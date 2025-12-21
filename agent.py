#!/usr/bin/env python3
"""
LangGraph Trading Agent

An AI trading agent built with LangGraph that thinks after each tool call.
Uses NASDAQ database and web search to make trading decisions.
"""

import json
import logging
import os
import signal
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Annotated, Dict, List, Optional

# Load environment variables
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

import config
import prompts
from memory_database import MemoryDatabase
from portfolio import Portfolio
from stock_history_database import StockHistoryDatabase

logger = logging.getLogger(__name__)


def extract_content_as_string(content) -> str:
    """
    Extract text content from message content that can be either string or list.
    GPT-5.2 and newer models can return content as a list of blocks.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        return "\n".join(text_parts)
    return str(content)  # Fallback for unexpected types


class StockScore(BaseModel):
    """Individual stock scoring breakdown."""

    symbol: str
    composite_score: float
    momentum_score: float
    quality_score: float
    technical_score: float


class ResearchOutput(BaseModel):
    """Structured output from Phase 1: Market Analysis."""

    current_holdings_scores: List[StockScore]  # Scores for current positions
    top_alternatives: List[StockScore]  # Top alternatives not held


class TradeRecommendation(BaseModel):
    """Structured trade recommendation from the trading agent."""

    action: str  # "BUY", "SELL", or "HOLD"
    symbol: str  # Stock symbol
    shares: int  # Number of shares (required for BUY/SELL, ignored for HOLD)
    reasoning: str  # Detailed reasoning for the recommendation
    confidence: Optional[str] = None  # "HIGH", "MEDIUM", "LOW"


class TradingOutput(BaseModel):
    """Complete trading analysis with optional recommendations."""

    complete_analysis: str  # Full AI analysis message from the agent
    summary: str  # Overall market analysis summary
    trade_recommendations: List[TradeRecommendation]  # Trade recommendations from agent
    market_outlook: str  # Bull/Bear/Neutral with reasoning
    risk_assessment: str  # Risk factors identified
    current_holdings_scores: List[StockScore]  # Scores for current positions (from Phase 1)
    top_alternatives: List[StockScore]  # Top alternatives not held (from Phase 1)


# Define the graph state
class TradingState(Dict):
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio_cash: float
    portfolio_positions: Dict[str, Dict]  # Serializable position data
    tool_call_count: int
    research_tool_calls: int  # Track research phase tool calls
    memory_tool_calls: int  # Track memory phase tool calls
    analysis_phase: str  # "research", "memory", or "trade"
    research_output: Optional[ResearchOutput] = None  # Scores from research phase
    trading_output: Optional[TradingOutput] = None  # Final output from trade phase


def _execute_sql_query(db_path: str, query: str) -> str:
    """
    Common SQL execution logic shared by run_sql and run_memory_sql tools.

    Args:
        db_path: Path to the SQLite database file
        query: SQL query to execute

    Returns:
        JSON string with query results or error
    """
    assert Path(db_path).exists(), f"Database file not found: {db_path}"
    assert query and query.strip(), "Empty query provided"

    def execute_query():
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return columns, rows

    # Execute query with timeout using ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_query)
            try:
                columns, rows = future.result(timeout=300)  # 5 minute timeout
            except FuturesTimeoutError:
                raise TimeoutError(f"SQL query exceeded 5 minute timeout: {query}")

        # Process results
        data = []
        for row in rows:
            row_dict = {}
            for i, value in enumerate(row):
                # Handle different data types
                if value is None:
                    row_dict[columns[i]] = None
                elif isinstance(value, (int, float, str)):
                    row_dict[columns[i]] = value
                else:
                    row_dict[columns[i]] = str(value)
            data.append(row_dict)

        result = {"columns": columns, "data": data, "row_count": len(data)}
    except Exception as e:
        logger.error(f"Error {e} executing query {query}")
        result = {"error": str(e), "query": query}
    return json.dumps(result, indent=2)


def create_run_sql_tool(cfg: config.Config):
    """Factory function to create run_sql tool with config context."""

    @tool
    def run_sql(query: str) -> str:
        """
        PRIMARY DATA GATHERING TOOL - Execute SQL queries against the NASDAQ stocks database.

        **CRITICAL: Use this tool extensively in Phase 1 (10-15 times minimum) before making any decisions.**

        This provides access to comprehensive NASDAQ 100 data:
        - 3 years of daily price history
        - Fundamental metrics (P/E, ROE, margins, debt, etc.)
        - Technical indicators (moving averages, volume, etc.)
        - Corporate actions (splits, dividends)

        Use this to:
        - Calculate momentum, returns, and performance metrics
        - Analyze fundamentals across stocks
        - Compare valuations and risk metrics
        - Identify trends and patterns

        Args:
            query: SQL SELECT query to execute (INSERT/UPDATE/DELETE not allowed)

        Returns:
            JSON string with query results, including data, columns, and metadata

        Example:
            run_sql("SELECT symbol, name, sector FROM stocks LIMIT 10")
            run_sql("SELECT symbol, close FROM stock_prices WHERE date = '2024-01-15'")
        """
        return _execute_sql_query(cfg.stock_history_db_name, query)

    return run_sql


def create_run_memory_sql_tool(cfg: config.Config):
    """Factory function to create run_memory_sql tool with config context."""

    @tool
    def run_memory_sql(query: str) -> str:
        """
        Query historical stock scores to understand multi-day trends before making trading decisions.

        Historical patterns reveal momentum shifts invisible in single-day data.

        Common patterns to analyze:
        - Compare holdings performance: SELECT symbol, AVG(composite_score) FROM agent_scores WHERE is_holding=1 AND date>=date('now','-7 days') GROUP BY symbol ORDER BY AVG(composite_score) DESC
        - Find strong alternatives: SELECT symbol, AVG(composite_score) FROM agent_scores WHERE is_holding=0 AND date>=date('now','-7 days') GROUP BY symbol HAVING AVG(composite_score)>=75 ORDER BY AVG(composite_score) DESC
        - Track specific stock: SELECT date, composite_score FROM agent_scores WHERE symbol='AAPL' AND date>=date('now','-7 days') ORDER BY date

        Args:
            query: SQL SELECT query to execute against memory database

        Returns:
            JSON string with query results, including data, columns, and metadata
        """
        return _execute_sql_query(cfg.memory_db_name, query)

    return run_memory_sql


def create_search_web_tool(cfg: config.Config):
    """Factory function to create search_web tool with config context."""

    @tool
    def search_web(query: str) -> str:
        """
        PRIMARY SENTIMENT & NEWS GATHERING TOOL - Search the web for market news, sentiment, and analysis.

        **CRITICAL: Use this tool extensively in Phase 1 (5-8 times minimum) before making any decisions.**

        This provides access to current market information not available in historical data:
        - Breaking news and recent earnings reports
        - Analyst opinions and market sentiment
        - Sector trends and macroeconomic factors
        - Company-specific developments (product launches, scandals, leadership changes)
        - Regulatory changes and industry shifts

        Use this to:
        - Check recent news for current holdings (e.g., "NVDA news December 2025")
        - Gauge market sentiment for top candidates (e.g., "GOOGL analyst ratings")
        - Identify sector trends (e.g., "semiconductor industry outlook")
        - Find red flags (e.g., "TSLA recall news", "META earnings miss")
        - Research macroeconomic factors (e.g., "Federal Reserve interest rate policy")

        **Best Practices:**
        - Include "news" for recent developments: "AAPL news"
        - Add time context when available: "MSFT earnings Q4 2025"
        - Search each current holding individually
        - Search top momentum stocks from your SQL queries
        - Search relevant sectors and broader market trends

        To find news, include "news" in your query (e.g., "NVDA news", "Federal Reserve latest news").

        Args:
            query: Search query (e.g., "NVDA earnings", "semiconductor sector outlook", "GOOGL news")
        """
        try:
            end_date = cfg.as_of_date if cfg.as_of_date else None

            if end_date:
                search = TavilySearch(max_results=3, kwargs={"end_date": end_date})
            else:
                search = TavilySearch(max_results=3)

            results = search.invoke(query)
            return json.dumps({"success": True, "query": query, "end_date": end_date, "results": results}, indent=2)
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "message": "Web search not available",
                }
            )

    return search_web


def create_analyze_stock_trends_tool(cfg: config.Config):
    """Factory function to create analyze_stock_trends tool with config context."""

    @tool
    def analyze_stock_trends(symbol: str, days: int = 14) -> str:
        """Analyze stock score trends, volatility, and sustained patterns for a specific stock.

        Args:
            symbol: Stock ticker to analyze (e.g. 'AAPL', 'GOOGL')
            days: Number of days to analyze (default 14)
        """
        try:
            memory_db = MemoryDatabase(cfg)
            analysis = memory_db.analyze_stock_trends(symbol, days)
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "symbol": symbol})

    return analyze_stock_trends


def create_compare_portfolio_performance_tool(cfg: config.Config):
    """Factory function to create compare_portfolio_performance tool with config context."""

    @tool
    def compare_portfolio_performance(days: int = 7) -> str:
        """Compare performance metrics across all current portfolio stocks.

        Args:
            days: Number of days to analyze (default 7)
        """
        try:
            memory_db = MemoryDatabase(cfg)
            comparison = memory_db.compare_portfolio_performance(days)
            return json.dumps(comparison, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return compare_portfolio_performance


def create_find_replacement_opportunities_tool(cfg: config.Config):
    """Factory function to create find_replacement_opportunities tool with config context."""

    @tool
    def find_replacement_opportunities(min_gap: float = 5.0, days: int = 7) -> str:
        """Find holdings that have clearly better alternatives available for strategic replacement.

        Args:
            min_gap: Minimum performance gap vs best alternative (default 5.0)
            days: Number of days to analyze (default 7)
        """
        try:
            memory_db = MemoryDatabase(cfg)
            opportunities = memory_db.find_replacement_opportunities(min_gap, days)
            return json.dumps(opportunities, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return find_replacement_opportunities


def create_find_stocks_to_sell_tool(cfg: config.Config):
    """Factory function to create find_stocks_to_sell tool with config context."""

    @tool
    def find_stocks_to_sell(days: int = 7, min_score_threshold: float = 60.0) -> str:
        """Find holdings that should be sold due to poor fundamental performance.

        Args:
            days: Number of days to analyze (default 7)
            min_score_threshold: Minimum score threshold below which stocks are candidates for selling (default 60.0)
        """
        try:
            memory_db = MemoryDatabase(cfg)
            sell_candidates = memory_db.find_stocks_to_sell(days, min_score_threshold)
            return json.dumps(sell_candidates, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return find_stocks_to_sell


def create_find_stocks_to_buy_tool(cfg: config.Config):
    """Factory function to create find_stocks_to_buy tool with config context."""

    @tool
    def find_stocks_to_buy(days: int = 7, min_score_threshold: float = 75.0, top_n: int = 10) -> str:
        """Find best available stocks (non-holdings) when cash is available.

        Args:
            days: Number of days to analyze (default 7)
            min_score_threshold: Minimum score threshold for buy candidates (default 75.0)
            top_n: Maximum number of buy candidates to return (default 10)
        """
        try:
            memory_db = MemoryDatabase(cfg)
            buy_candidates = memory_db.find_stocks_to_buy(days, min_score_threshold, top_n)
            return json.dumps(buy_candidates, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return find_stocks_to_buy


def create_get_confidence_metrics_tool(cfg: config.Config):
    """Factory function to create get_confidence_metrics tool with config context."""

    @tool
    def get_confidence_metrics(symbol: str) -> str:
        """Get comprehensive confidence metrics for trading decisions.

        Args:
            symbol: Optional stock ticker to focus on, or None for overall metrics
        """
        try:
            memory_db = MemoryDatabase(cfg)
            metrics = memory_db.get_confidence_metrics(symbol)
            return json.dumps(metrics, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "symbol": symbol})

    return get_confidence_metrics


# Define the agent nodes
def initialize_agent_node(state: TradingState, cfg: config.Config) -> TradingState:
    """Initialize agent with system prompt and market analysis prompt."""
    # Load portfolio to get prices_as_of for the prompts
    assert os.path.exists(cfg.portfolio_file), f"Portfolio file not found: {cfg.portfolio_file}"
    portfolio = Portfolio.load(cfg)

    # Add system message introducing the agent and its goal
    system_msg = prompts.get_system_prompt(
        portfolio_cash=state["portfolio_cash"],
        portfolio_positions=state["portfolio_positions"],
        data_as_of_date=portfolio.prices_as_of or "unknown",
        cfg=cfg,
    )

    # Add research phase prompt
    research_prompt = prompts.get_research_prompt(
        portfolio_cash=state["portfolio_cash"],
        portfolio_positions=state["portfolio_positions"],
        data_as_of_date=portfolio.prices_as_of or "unknown",
        cfg=cfg,
    )

    # Add system message and research prompt
    state["messages"].append(SystemMessage(content=system_msg))
    state["messages"].append(HumanMessage(content=research_prompt))

    return state


def create_research_node(llm_with_tools, cfg: config.Config):
    """Create the research node with the configured LLM."""

    def research_node(state: TradingState) -> TradingState:
        """Analyze market data using run_sql and search_web."""
        # The LLM will respond with tool calls, which will be handled by the tool node
        response = llm_with_tools.invoke(state["messages"])
        logger.info(f"Research node reasoning: {response.content}")
        state["messages"].append(response)
        return state

    return research_node


def create_research_output_node(structured_llm, cfg: config.Config):
    """Create node to extract structured scores from research phase."""

    def research_output_node(state: TradingState) -> TradingState:
        """Extract structured scores from research phase."""
        scores_prompt = f"""Based on your research, provide structured scores.

Provide scores for:
1. ALL current holdings
2. TOP {cfg.top_alternatives_count} promising alternatives you identified

Use the scores you calculated during your analysis."""

        state["messages"].append(HumanMessage(content=scores_prompt))
        research_output = structured_llm.invoke(state["messages"])
        state["research_output"] = research_output

        logger.info(f"Research scores captured: {len(research_output.current_holdings_scores)} holdings, {len(research_output.top_alternatives)} alternatives")

        # Add human-readable scores to conversation for visibility
        scores_text = "Research phase complete. Scores extracted:\n\n"

        scores_text += "**Current Holdings:**\n"
        for score in research_output.current_holdings_scores:
            scores_text += f"- {score.symbol}: Composite={score.composite_score:.1f}, Momentum={score.momentum_score:.1f}, Quality={score.quality_score:.1f}, Technical={score.technical_score:.1f}\n"

        scores_text += "\n**Top Alternatives:**\n"
        for score in research_output.top_alternatives:
            scores_text += f"- {score.symbol}: Composite={score.composite_score:.1f}, Momentum={score.momentum_score:.1f}, Quality={score.quality_score:.1f}, Technical={score.technical_score:.1f}\n"

        state["messages"].append(AIMessage(content=scores_text))

        return state

    return research_output_node


def create_memory_node(llm_with_tools, cfg: config.Config):
    """Create the memory analysis node with the configured LLM."""

    def memory_node(state: TradingState) -> TradingState:
        """Analyze historical patterns using memory tools."""
        # Add transition message when entering memory phase
        if state["analysis_phase"] == "research":
            state["analysis_phase"] = "memory"
            memory_prompt = prompts.get_memory_prompt(
                portfolio_cash=state["portfolio_cash"],
                portfolio_positions=state["portfolio_positions"],
                research_output=state["research_output"],
                cfg=cfg,
            )
            state["messages"].append(HumanMessage(content=memory_prompt))

        # The LLM will respond with tool calls, which will be handled by the memory tool node
        response = llm_with_tools.invoke(state["messages"])
        logger.info(f"Memory analysis node reasoning: {response.content}")
        state["messages"].append(response)
        return state

    return memory_node


def create_trade_node(llm, cfg: config.Config):
    """Create the trade node for Phase 3 reasoning."""

    def trade_node(state: TradingState) -> TradingState:
        """Reason about trading decisions using research scores and memory patterns."""
        # Add transition message when entering trade phase
        if state["analysis_phase"] == "memory":
            state["analysis_phase"] = "trade"

            # Collect analysis context from previous messages
            analysis_context = ""
            for msg in state["messages"]:
                if isinstance(msg, AIMessage) and msg.content:
                    analysis_context += extract_content_as_string(msg.content) + "\n\n"

            trade_prompt = prompts.get_trade_prompt(
                portfolio_cash=state["portfolio_cash"],
                portfolio_positions=state["portfolio_positions"],
                analysis_context=analysis_context,
                cfg=cfg,
            )
            state["messages"].append(HumanMessage(content=trade_prompt))

        # The LLM reasons about trades (no tools)
        response = llm.invoke(state["messages"])
        logger.info(f"Trade node reasoning: {response.content}")
        state["messages"].append(response)
        return state

    return trade_node


def create_trading_output_node(structured_llm, cfg: config.Config):
    """Create the trading output node to extract structured recommendations."""

    def trading_output_node(state: TradingState) -> TradingState:
        """Extract structured trading output."""
        output_prompt = """Based on your trading analysis, provide structured output with:
1. Summary of your market analysis
2. Specific trade recommendations (BUY/SELL/HOLD with shares and reasoning)
3. Current holdings scores from Phase 1
4. Top alternatives scores from Phase 1
5. Market outlook
6. Risk assessment"""

        state["messages"].append(HumanMessage(content=output_prompt))
        trading_analysis = structured_llm.invoke(state["messages"])

        # Set the complete analysis from the previous AI message (before the output_prompt)
        # Get the second-to-last message which has the trade reasoning
        if len(state["messages"]) >= 2:
            trade_message = state["messages"][-2]
            if isinstance(trade_message, AIMessage) and trade_message.content:
                trading_analysis.complete_analysis = extract_content_as_string(trade_message.content)

        # Store the structured analysis in state for later use
        state["trading_output"] = trading_analysis

        analysis_text = print_analysis(trading_analysis)
        state["messages"].append(AIMessage(content=analysis_text))
        return state

    return trading_output_node


def create_tools_node_wrapper(tool_node, cfg: config.Config, phase: str):
    """Create the tools node wrapper with the configured tool node.

    Args:
        tool_node: The ToolNode to execute
        cfg: Configuration object
        phase: "research" or "memory" to track phase-specific tool calls
    """

    def tools_node_wrapper(state: TradingState) -> TradingState:
        """Execute tools and increment the phase-specific tool call counter."""
        # Count how many tool calls we're about to make
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            state["tool_call_count"] += 1
            if phase == "research":
                state["research_tool_calls"] += 1
            elif phase == "memory":
                state["memory_tool_calls"] += 1

        # Execute the tools
        result = tool_node.invoke(state)

        # Ensure the counters are preserved in the result
        result["tool_call_count"] = state["tool_call_count"]
        result["research_tool_calls"] = state["research_tool_calls"]
        result["memory_tool_calls"] = state["memory_tool_calls"]
        result["analysis_phase"] = state["analysis_phase"]
        return result

    return tools_node_wrapper


# Define routing logic
def should_continue_research(state: TradingState, cfg: config.Config):
    """Route decision for research phase."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, and we haven't reached the tool call limit, go to research_tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["research_tool_calls"] < cfg.max_tool_calls:
        logger.info(f"""Research phase tool call {[tool["name"] for tool in last_message.tool_calls]} count: {state["research_tool_calls"]}""")
        return "research_tools"
    # Otherwise research phase complete, extract structured scores
    logger.info("Research phase complete, extracting structured scores")
    return "research_output"


def should_continue_memory(state: TradingState, cfg: config.Config):
    """Route decision for memory phase."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, and we haven't reached the tool call limit, go to memory_tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["memory_tool_calls"] < cfg.max_tool_calls:
        logger.info(f"""Memory phase tool call {[tool["name"] for tool in last_message.tool_calls]} count: {state["memory_tool_calls"]}""")
        return "memory_tools"
    # Otherwise memory phase complete, move to trade phase for reasoning
    logger.info("Memory phase complete, moving to trade phase")
    return "trade"


def print_analysis(trading_analysis, use_markdown: bool = False):
    # Complete Analysis
    if use_markdown:
        text = f"## Complete Analysis\n{trading_analysis.complete_analysis}\n\n"
        text += f"## Market Analysis Summary\n{trading_analysis.summary}\n\n"
    else:
        text = f"""
🤖 COMPLETE ANALYSIS:
{trading_analysis.complete_analysis}

🎯 TRADING ANALYSIS COMPLETE
📊 Market Analysis Summary: {trading_analysis.summary}
🎯 Market Outlook: {trading_analysis.market_outlook}"""

    # Risk Assessment
    if use_markdown:
        text += f"## Risk Assessment\n{trading_analysis.risk_assessment}\n\n"
    else:
        text += f"\n⚠️ Risk Assessment: {trading_analysis.risk_assessment}"

    # Trade Recommendations
    if trading_analysis.trade_recommendations:
        if use_markdown:
            text += "## Trade Recommendations\n\n"
            text += "| Action | Symbol | Shares | Confidence | Reasoning |\n"
            text += "|--------|--------|-------:|-----------:|:----------|\n"
            for rec in trading_analysis.trade_recommendations:
                text += f"| **{rec.action}** | {rec.symbol} | {rec.shares:,} | {rec.confidence} | {rec.reasoning} |\n"
            text += "\n"
        else:
            text += "\n📋 Trade Recommendations\n"
            for rec in trading_analysis.trade_recommendations:
                text += f"{str(rec)}\n"

    # Current Holdings Scores
    if use_markdown:
        text += "## Stock Scores - Current Holdings\n\n"
        text += "| Symbol | Composite | Momentum | Quality | Technical |\n"
        text += "|--------|----------:|---------:|--------:|----------:|\n"
        for rec in trading_analysis.current_holdings_scores:
            text += f"| {rec.symbol} | {rec.composite_score:.1f} | {rec.momentum_score:.1f} | {rec.quality_score:.1f} | {rec.technical_score:.1f} |\n"
        text += "\n"
    else:
        text += "📋 Current Holdings Scores:\n"
        for rec in trading_analysis.current_holdings_scores:
            text += f"{rec.symbol}: {rec.composite_score}\n"

    # Top Alternatives
    if use_markdown:
        text += "## Stock Scores - Top Alternatives\n\n"
        text += "| Symbol | Composite | Momentum | Quality | Technical |\n"
        text += "|--------|----------:|---------:|--------:|----------:|\n"
        for rec in trading_analysis.top_alternatives:
            text += f"| {rec.symbol} | {rec.composite_score:.1f} | {rec.momentum_score:.1f} | {rec.quality_score:.1f} | {rec.technical_score:.1f} |\n"
        text += "\n"
    else:
        text += "📋 Top Alternatives:\n"
        for rec in trading_analysis.top_alternatives:
            text += f"{rec.symbol}: {rec.composite_score}\n"

    # Market Outlook
    if use_markdown:
        text += f"## Market Outlook\n**{trading_analysis.market_outlook}**\n"
    else:
        text += "✅ Analysis session completed!"

    return text


def main(cfg: config.Config | None = None):
    """Main application entry point"""
    # Load environment variables
    load_dotenv()

    # Parse configuration with command line overrides (if not provided)
    if cfg is None:
        cfg = config.parse_config()

    # Setup logging
    config.setup_logging()

    if "gpt" in cfg.llm_model:
        # Configure the LLM with deterministic settings
        if "pro" in cfg.llm_model:
            llm = ChatOpenAI(
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
            )
        else:
            llm = ChatOpenAI(
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
                seed=cfg.llm_seed,
            )
    elif "gemini" in cfg.llm_model:
        llm = ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            convert_system_message_to_human=True,
        )
    else:
        raise ValueError(f"Unsupported LLM model: {cfg.llm_model}")

    # Create structured LLMs for different output types
    research_structured_llm = llm.with_structured_output(ResearchOutput)
    trading_structured_llm = llm.with_structured_output(TradingOutput)

    # Tool setup - split into research and memory tools
    # Research tools: data gathering from stock database and web
    research_tools = [
        create_run_sql_tool(cfg),
        create_search_web_tool(cfg),
    ]

    # Memory tools: analytical tools + SQL for flexibility
    memory_tools = [
        create_analyze_stock_trends_tool(cfg),
        create_compare_portfolio_performance_tool(cfg),
        create_run_memory_sql_tool(cfg),
    ]

    # Create separate tool nodes and LLMs for each phase
    research_tool_node = ToolNode(research_tools)
    memory_tool_node = ToolNode(memory_tools)
    llm_with_research_tools = llm.bind_tools(research_tools)
    llm_with_memory_tools = llm.bind_tools(memory_tools)

    # Build the graph
    workflow = StateGraph(TradingState)

    # Create node functions with dependencies
    research_node = create_research_node(llm_with_research_tools, cfg)
    research_output_node = create_research_output_node(research_structured_llm, cfg)
    research_tools_wrapper = create_tools_node_wrapper(research_tool_node, cfg, "research")
    memory_node = create_memory_node(llm_with_memory_tools, cfg)
    memory_tools_wrapper = create_tools_node_wrapper(memory_tool_node, cfg, "memory")
    trade_node = create_trade_node(llm, cfg)
    trading_output_node = create_trading_output_node(trading_structured_llm, cfg)

    # Add nodes
    workflow.add_node("initialize_agent", lambda state: initialize_agent_node(state, cfg))
    workflow.add_node("research", research_node)
    workflow.add_node("research_tools", research_tools_wrapper)
    workflow.add_node("research_output", research_output_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("memory_tools", memory_tools_wrapper)
    workflow.add_node("trade", trade_node)
    workflow.add_node("trading_output", trading_output_node)

    # Add edges
    workflow.set_entry_point("initialize_agent")
    workflow.add_edge("initialize_agent", "research")

    # Research phase: loops between research and research_tools
    # Routes to either "research_tools" (continue) or "research_output" (done with research phase)
    workflow.add_conditional_edges("research", lambda state: should_continue_research(state, cfg))
    workflow.add_edge("research_tools", "research")

    # Extract structured scores from research phase, then move to memory phase
    workflow.add_edge("research_output", "memory")

    # Memory phase: loops between memory and memory_tools
    # Routes to either "memory_tools" (continue) or "trade" (done with memory phase)
    workflow.add_conditional_edges("memory", lambda state: should_continue_memory(state, cfg))
    workflow.add_edge("memory_tools", "memory")

    # Trade phase: reason about trades, then extract structured output
    workflow.add_edge("trade", "trading_output")

    # Final output
    workflow.add_edge("trading_output", END)

    # Compile the graph
    app = workflow.compile()

    # Run the LangGraph trading agent
    logger.info("🚀 Starting LangGraph Trading Agent...")
    logger.info("=" * 80)

    # Load portfolio and print initial state
    portfolio = Portfolio.load(cfg)
    logger.info(portfolio.print("Initial Portfolio"))

    # Initialize state with values from portfolio
    initial_state = TradingState(
        messages=[],
        portfolio_cash=portfolio.cash,
        portfolio_positions=portfolio.positions,
        tool_call_count=0,
        research_tool_calls=0,
        memory_tool_calls=0,
        analysis_phase="research",
        research_output=None,
        trading_output=None,
    )

    final_state = None

    # Set up timeout handler
    def timeout_handler(_signum, _frame):
        raise TimeoutError("LangGraph execution exceeded timeout")

    # Set alarm for 30 minutes (1800 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800)

    try:
        # Run the graph
        for step in app.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, state in step.items():
                final_state = state  # Keep track of final state
                if node_name != "tools":  # Don't print tool outputs directly
                    logger.info(f"\n--- {node_name.upper()} ---")
                    # if state["messages"]:
                    #     last_msg = state["messages"][-1]
                    #     logger.debug(extract_content_as_string(last_msg.content))

        logger.info("\n✅ Trading session completed successfully!")
    except TimeoutError:
        logger.error("❌ LangGraph execution timed out after 30 minutes")
        raise
    finally:
        # Cancel the alarm
        signal.alarm(0)

    # Return structured output directly
    assert final_state, "Final state must exist"
    trading_output = final_state["trading_output"]
    assert trading_output, "Structured analysis must be present in final state"
    logger.info(print_analysis(trading_output))

    # Log final portfolio comparison
    logger.info(portfolio.print("Current Portfolio"))

    # Only log "after" portfolio if there are actual trade recommendations
    # there are always trade recommendations
    assert trading_output.trade_recommendations, "Trade recommendations must be present in final state"

    # Apply trades and show resulting portfolio
    final_portfolio = portfolio.apply_trades(trading_output.trade_recommendations)
    if final_portfolio:
        logger.info(final_portfolio.print("Portfolio After Trades"))
        if cfg.execute_trades:
            final_portfolio.save()
        else:
            logger.info("📋 Trade execution disabled - portfolio file not updated")

    # Validate and filter scores before saving to memory
    stock_db = StockHistoryDatabase(cfg)

    prices = {}
    # Validate holdings - all should exist since they're actual positions
    validated_holdings = []
    for score in trading_output.current_holdings_scores:
        try:
            prices[score.symbol] = stock_db.get_latest_price(score.symbol)
            validated_holdings.append(score)
        except AssertionError:
            logger.warning(f"Skipping holding score for {score.symbol} - not found in database")

    # Validate alternatives - filter out hallucinations
    validated_alternatives = []
    for score in trading_output.top_alternatives:
        try:
            prices[score.symbol] = stock_db.get_latest_price(score.symbol)
            validated_alternatives.append(score)
        except AssertionError:
            logger.warning(f"Skipping alternative {score.symbol} - not found in database (possible hallucination)")

    # Save validated scores to memory (use same date as portfolio snapshot)
    memory_db = MemoryDatabase(cfg)
    memory_date = portfolio.prices_as_of
    memory_db.update_memory(memory_date, validated_holdings, validated_alternatives, prices)

    return trading_output, final_portfolio


if __name__ == "__main__":
    main()
