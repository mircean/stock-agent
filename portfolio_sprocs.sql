-- Portfolio Stored Procedures
-- Operations for portfolio history tracking

-- Function to get portfolio history
CREATE OR REPLACE FUNCTION get_portfolio_history(days_limit INTEGER DEFAULT 30)
RETURNS TABLE (
    date DATE,
    cash NUMERIC(12,4),
    positions_value NUMERIC(12,4),
    total_value NUMERIC(12,4),
    day_change NUMERIC(12,4),
    day_change_pct NUMERIC(6,2),
    position_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ph.date,
        ph.cash,
        ph.positions_value,
        ph.total_value,
        ph.day_change,
        ph.day_change_pct,
        ph.position_count
    FROM portfolio_history ph
    ORDER BY ph.date DESC
    LIMIT days_limit;
END;
$$ LANGUAGE plpgsql;

