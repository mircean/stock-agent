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
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import config
import prompts

# Load environment variables
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from memory_database import MemoryDatabase
from portfolio import Portfolio
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TradeRecommendation(BaseModel):
    """Structured trade recommendation from the trading agent."""

    action: str  # "BUY", "SELL", or "HOLD"
    symbol: Optional[str] = None  # Stock symbol (if applicable)
    shares: Optional[float] = None  # Number of shares
    price: Optional[float] = None  # Target price
    reasoning: str  # Detailed reasoning for the recommendation
    confidence: Optional[str] = None  # "HIGH", "MEDIUM", "LOW"


class StockScore(BaseModel):
    """Individual stock scoring breakdown."""

    symbol: str
    composite_score: float
    momentum_score: float
    quality_score: float
    technical_score: float
    current_price: float


class TradingAnalysis(BaseModel):
    """Complete trading analysis with optional recommendations."""

    complete_analysis: str  # Full AI analysis message from the agent
    summary: str  # Overall market analysis summary
    trade_recommendations: List[TradeRecommendation] = []  # Trade recommendations from agent
    market_outlook: str  # Bull/Bear/Neutral with reasoning
    risk_assessment: str  # Risk factors identified
    current_holdings_scores: List[StockScore]  # Scores for current positions
    top_alternatives: List[StockScore]  # Top alternatives not held


# Define the graph state
class TradingState(Dict):
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio_cash: float
    portfolio_positions: Dict[str, Dict]  # Serializable position data
    tool_call_count: int
    trading_analysis: Optional[TradingAnalysis] = None  # Structured output from LLM


def create_run_sql_tool(cfg: config.Config):
    """Factory function to create run_sql tool with config context."""

    @tool
    def run_sql(query: str) -> str:
        """
        Execute a SQL query against the NASDAQ stocks database.

        This function provides access to a comprehensive database of NASDAQ 100 stocks
        with 3 years of historical data including prices, fundamentals, and statistics.

        Args:
            query: SQL SELECT query to execute (INSERT/UPDATE/DELETE not allowed)

        Returns:
            JSON string with query results, including data, columns, and metadata

        Example:
            run_sql("SELECT symbol, name, sector FROM stocks LIMIT 10")
            run_sql("SELECT symbol, close FROM stock_prices WHERE date = '2024-01-15'")
        """

        db_path = cfg.stock_history_db_name
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
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_query)
            try:
                columns, rows = future.result(timeout=300)  # 5 minute timeout
            except FuturesTimeoutError:
                raise TimeoutError(f"SQL query exceeded 5 minute timeout: {query[:100]}")

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
        return json.dumps(result, indent=2)

    return run_sql


def create_search_web_tool(cfg: config.Config):
    """Factory function to create search_web tool with config context."""

    @tool
    def search_web(query: str) -> str:
        """
        Search the web for market information and analysis.

        Can search for news articles, analyst reports, sector trends, or general information.
        To find news, include "news" in your query (e.g., "NVDA news", "Federal Reserve latest news").

        Args:
            query: Search query (e.g., "NVDA earnings", "semiconductor sector outlook", "GOOGL news")
        """
        try:
            end_date = cfg.as_of_date if cfg.as_of_date else None

            if end_date:
                search = TavilySearchResults(max_results=3, kwargs={"end_date": end_date})
            else:
                search = TavilySearchResults(max_results=3)

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
    def get_confidence_metrics(symbol: str = None) -> str:
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
    """Initialize agent with portfolio and system message."""
    assert os.path.exists(cfg.portfolio_file), f"Portfolio file not found: {cfg.portfolio_file}"
    # Load portfolio data with Lot objects
    portfolio = Portfolio.load(cfg)

    state["portfolio_cash"] = portfolio.cash
    state["portfolio_positions"] = portfolio.positions
    state["tool_call_count"] = 0

    # Add system message about portfolio and constraints
    system_msg = prompts.get_system_prompt(
        portfolio_cash=state["portfolio_cash"],
        portfolio_positions=state["portfolio_positions"],
        cfg=cfg,
    )

    state["messages"].append(SystemMessage(content=system_msg))

    return state


def create_market_analysis_node(llm_with_tools, cfg: config.Config):
    """Create the market analysis node with the configured LLM."""

    def market_analysis_node(state: TradingState) -> TradingState:
        """Analyze market data and make trading decisions."""
        # The LLM will respond with tool calls, which will be handled by the tool node
        response = llm_with_tools.invoke(state["messages"])
        logger.info(f"Market analysis node reasoning: {response.content}")
        state["messages"].append(response)
        return state

    return market_analysis_node


def create_tools_node_wrapper(tool_node, cfg: config.Config):
    """Create the tools node wrapper with the configured tool node."""

    def tools_node_wrapper(state: TradingState) -> TradingState:
        """Execute tools and increment the tool call counter."""
        # Count how many tool calls we're about to make, or how many iterations so far
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            state["tool_call_count"] += 1  # len(last_message.tool_calls)

        # Execute the tools
        result = tool_node.invoke(state)

        # Ensure the counter is preserved in the result
        result["tool_call_count"] = state["tool_call_count"]
        return result

    return tools_node_wrapper


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
            text += "| Action | Symbol | Shares | Price | Confidence | Reasoning |\n"
            text += "|--------|--------|-------:|------:|-----------:|:----------|\n"
            for rec in trading_analysis.trade_recommendations:
                price_str = f"${rec.price:.2f}" if rec.price else "Market"
                shares_str = f"{rec.shares:,}" if rec.shares else "N/A"
                symbol_str = rec.symbol if rec.symbol else "N/A"
                text += f"| **{rec.action}** | {symbol_str} | {shares_str} | {price_str} | {rec.confidence} | {rec.reasoning} |\n"
            text += "\n"
        else:
            text += "\n📋 Trade Recommendations\n"
            for rec in trading_analysis.trade_recommendations:
                text += f"{str(rec)}\n"

    # Current Holdings Scores
    if use_markdown:
        text += "## Stock Scores - Current Holdings\n\n"
        text += "| Symbol | Composite | Momentum | Quality | Technical | Price |\n"
        text += "|--------|----------:|---------:|--------:|----------:|------:|\n"
        for rec in trading_analysis.current_holdings_scores:
            text += f"| {rec.symbol} | {rec.composite_score:.1f} | {rec.momentum_score:.1f} | {rec.quality_score:.1f} | {rec.technical_score:.1f} | ${rec.current_price:.2f} |\n"
        text += "\n"
    else:
        text += "📋 Current Holdings Scores:\n"
        for rec in trading_analysis.current_holdings_scores:
            text += f"{rec.symbol}: {rec.composite_score}\n"

    # Top Alternatives
    if use_markdown:
        text += "## Stock Scores - Top Alternatives\n\n"
        text += "| Symbol | Composite | Momentum | Quality | Technical | Price |\n"
        text += "|--------|----------:|---------:|--------:|----------:|------:|\n"
        for rec in trading_analysis.top_alternatives:
            text += f"| {rec.symbol} | {rec.composite_score:.1f} | {rec.momentum_score:.1f} | {rec.quality_score:.1f} | {rec.technical_score:.1f} | ${rec.current_price:.2f} |\n"
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


def create_analysis_output_node(structured_llm, cfg: config.Config):
    """Create the analysis output node with the configured structured LLM."""

    def analysis_output_node(state: TradingState) -> TradingState:
        """Provide final structured analysis output (scores only or with recommendations)."""
        # Collect analysis context from previous messages
        analysis_context = ""
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                analysis_context += msg.content + "\n\n"

        # Get structured analysis prompt
        structured_prompt = prompts.get_trading_analysis_prompt(
            portfolio_cash=state["portfolio_cash"],
            portfolio_positions=state["portfolio_positions"],
            analysis_context=analysis_context[-2000:],  # Limit context length
            cfg=cfg,
        )

        # Get structured output from LLM
        trading_analysis = structured_llm.invoke([HumanMessage(content=structured_prompt)])

        # Set the complete analysis from the last AI message
        last_message = state["messages"][-1]
        assert isinstance(last_message, AIMessage) and last_message.content, "Last message must be an AIMessage with content"
        trading_analysis.complete_analysis = last_message.content

        # Store the structured analysis in state for later use
        state["trading_analysis"] = trading_analysis

        analysis_text = print_analysis(trading_analysis)
        # Create a human-readable summary message

        state["messages"].append(AIMessage(content=analysis_text))
        return state

    return analysis_output_node


# Define routing logic
def should_continue(state: TradingState, cfg: config.Config):
    last_message = state["messages"][-1]

    # If the last message has tool calls, and we haven't reached the tool call limit, go to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["tool_call_count"] < cfg.max_tool_calls:
        logger.info(f"""Tool call {[tool["name"] for tool in last_message.tool_calls]} tool call count: {state["tool_call_count"]}""")
        return "tools"
    # Otherwise go to analysis_output (analysis is complete)
    return "analysis_output"


def main(cfg: config.Config = None):
    """Main application entry point"""
    # Load environment variables
    load_dotenv()

    # Parse configuration with command line overrides (if not provided)
    if cfg is None:
        cfg = config.parse_config()

    # Setup logging
    config.setup_logging()

    # Configure the LLM with deterministic settings
    llm = ChatOpenAI(model=cfg.llm_model, temperature=cfg.llm_temperature, seed=cfg.llm_seed)

    # Always use TradingAnalysis (trade_recommendations will be empty in scores-only mode)
    structured_llm = llm.with_structured_output(TradingAnalysis)

    # Tool setup - create all tools with config context
    run_sql = create_run_sql_tool(cfg)
    search_web = create_search_web_tool(cfg)
    analyze_stock_trends = create_analyze_stock_trends_tool(cfg)
    compare_portfolio_performance = create_compare_portfolio_performance_tool(cfg)
    find_replacement_opportunities = create_find_replacement_opportunities_tool(cfg)
    find_stocks_to_sell = create_find_stocks_to_sell_tool(cfg)
    find_stocks_to_buy = create_find_stocks_to_buy_tool(cfg)
    get_confidence_metrics = create_get_confidence_metrics_tool(cfg)

    tools = [
        run_sql,
        search_web,
        analyze_stock_trends,
        compare_portfolio_performance,
        find_replacement_opportunities,
        find_stocks_to_sell,
        find_stocks_to_buy,
        get_confidence_metrics,
    ]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    # Build the graph
    workflow = StateGraph(TradingState)

    # Create node functions with dependencies
    market_analysis_node = create_market_analysis_node(llm_with_tools, cfg)
    tools_node_wrapper = create_tools_node_wrapper(tool_node, cfg)
    analysis_output_node = create_analysis_output_node(structured_llm, cfg)

    # Add nodes
    workflow.add_node("initialize_agent", lambda state: initialize_agent_node(state, cfg))
    workflow.add_node("market_analysis", market_analysis_node)
    workflow.add_node("analysis_output", analysis_output_node)
    workflow.add_node("tools", tools_node_wrapper)

    # Add edges
    workflow.set_entry_point("initialize_agent")
    workflow.add_edge("initialize_agent", "market_analysis")
    workflow.add_conditional_edges("market_analysis", lambda state: should_continue(state, cfg))
    workflow.add_edge("tools", "market_analysis")
    workflow.add_edge("analysis_output", END)

    # Compile the graph
    app = workflow.compile()

    # Run the LangGraph trading agent
    logger.info("🚀 Starting LangGraph Trading Agent...")
    logger.info("=" * 80)

    # Initialize state
    initial_state = TradingState(
        messages=[],
        portfolio_cash=cfg.default_cash,
        portfolio_positions={},
        tool_call_count=0,
        trading_analysis=None,
    )

    # Print initial portfolio
    portfolio = Portfolio.load(cfg)
    logger.info(portfolio.print("Initial Portfolio"))

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
                    if state["messages"]:
                        last_msg = state["messages"][-1]
                        logger.debug(last_msg.content)

        logger.info("\n✅ Trading session completed successfully!")
    except TimeoutError:
        logger.error("❌ LangGraph execution timed out after 30 minutes")
        raise
    finally:
        # Cancel the alarm
        signal.alarm(0)

    # Return structured output directly
    assert final_state, "Final state must exist"
    trading_analysis = final_state["trading_analysis"]
    assert trading_analysis, "Structured analysis must be present in final state"
    logger.info(print_analysis(trading_analysis))

    # Log final portfolio comparison
    logger.info(portfolio.print("Current Portfolio"))

    # Only log "after" portfolio if there are actual trade recommendations
    # there are always trade recommendations
    assert trading_analysis.trade_recommendations, "Trade recommendations must be present in final state"

    # Apply trades and show resulting portfolio
    final_portfolio = portfolio.apply_trades(trading_analysis.trade_recommendations)
    if final_portfolio:
        logger.info(final_portfolio.print("Portfolio After Trades"))
        if cfg.execute_trades:
            final_portfolio.save()
        else:
            logger.info("📋 Trade execution disabled - portfolio file not updated")

    # Save scores to memory (use same date as portfolio snapshot)
    memory_db = MemoryDatabase(cfg)
    memory_date = portfolio.prices_as_of if portfolio.prices_as_of else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_db.update_memory(memory_date, trading_analysis.current_holdings_scores, trading_analysis.top_alternatives)

    return trading_analysis, final_portfolio


if __name__ == "__main__":
    main()
