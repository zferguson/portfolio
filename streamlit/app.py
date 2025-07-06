import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.portfolio_metrics import portfolio_performance, optimize_portfolio
from data.load_data import get_market_data, TICKERS, BENCHMARK


st.set_page_config(
    page_title="Portfolio", 
    layout="wide"
)

st.title("Portfolio Dashboard")
st.markdown("""
    Welcome to my portfolio dashboard! Use the controls in the sidebar to select different stocks, benchmarks, and time periods.
    """)


# --- Session State Management ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'ticker_list' not in st.session_state:
    st.session_state['ticker_list'] = TICKERS
if 'benchmark' not in st.session_state:
    st.session_state['benchmark'] = BENCHMARK
if 'time_period' not in st.session_state:
    st.session_state['time_period'] = '5y'
if 'time_interval' not in st.session_state:
    st.session_state['time_interval'] = '1d'

def fetch_market_data():
    st.session_state['data'] = get_market_data(
        tickers=st.session_state['ticker_list'],
        benchmark=st.session_state['benchmark'],
        period=st.session_state['time_period'],
        interval=st.session_state['time_interval']
    )

# Sidebar controls update session state, but do not fetch data

# --- Sidebar ---
with st.sidebar:
    st.header(":gear: Controls")

    st.session_state['ticker_list'] = st.multiselect(
        "Select Tickers",
        options=TICKERS,
        default=st.session_state['ticker_list']
    )

    st.session_state['time_period'] = st.selectbox(
        "Select Time Period",
        options=['1y', '3y', '5y', '10y'],
        index=['1y', '3y', '5y', '10y'].index(st.session_state['time_period'])
    )

    st.session_state['time_interval'] = st.selectbox(
        "Select Time Interval",
        options=['1d', '1wk', '1mo'],
        index=['1d', '1wk', '1mo'].index(st.session_state['time_interval'])
    )

    st.session_state['benchmark'] = st.selectbox(
        "Select Benchmark",
        options=['SPY', 'QQQ', 'IWM', 'VTI'],
        index=['SPY', 'QQQ', 'IWM', 'VTI'].index(st.session_state['benchmark'])
    )

    fetch_data = st.button(
        "Fetch Market Data",
        use_container_width=True,
        type="primary",
        on_click=fetch_market_data
    )

# --- Main App ---
data = st.session_state['data']
ticker_list = st.session_state['ticker_list']
benchmark = st.session_state['benchmark']
time_period = st.session_state['time_period']
time_interval = st.session_state['time_interval']

if data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Prices", "Returns", "Ratios", "Efficient Frontier"])

    with tab1:
        st.header("Prices")
        st.dataframe(data['prices'], use_container_width=True)

    with tab2:
        st.header("Returns")
        st.dataframe(data['returns'], use_container_width=True)

    with tab3:
        st.header("Ratios")
        # Calculate annualized return and risk for each security
        prices = data['prices']
        returns = prices.pct_change().dropna()
        periods_per_year = {'1d': 252, '1wk': 52, '1mo': 12}.get(time_interval, 252)
        ann_return = (1 + returns.mean()) ** periods_per_year - 1
        ann_risk = returns.std() * np.sqrt(periods_per_year)
        ratios_df = data['ratios'].copy()
        ratios_df['Annualized Return'] = ann_return
        ratios_df['Annualized Risk'] = ann_risk
        st.dataframe(ratios_df, use_container_width=True)

    with tab4:
        st.header("Efficient Frontier")
        st.markdown("""
            The efficient frontier is a concept in modern portfolio theory that represents the set of optimal portfolios that offer the highest expected return for a given level of risk.
        """)

        # User selects metric for optimization
        metric_options = [
            'sharpe_ratio', 'sortino_ratio', 'annualized_return', 'annualized_risk', 'max_drawdown', 'tail_risk_5pct'
        ]
        selected_metric = st.selectbox(
            "Select metric to optimize for optimal portfolio:",
            options=metric_options,
            index=0,
            help="Choose which metric to optimize for the optimal portfolio weights."
        )

        # Only show optimal if enough tickers
        if len(ticker_list) > 1:
            opt_result = optimize_portfolio(
                data['prices'],
                metric=selected_metric,
                benchmark=benchmark
            )
            opt_weights = opt_result['optimal_weights']

        # Customizable weights table
        default_weight = round(100 / len(ticker_list), 2) if ticker_list else 0
        weights_df = pd.DataFrame({
            "Ticker": ticker_list,
            "Custom Weight (%)": [default_weight] * len(ticker_list),
            "Optimal Weight (%)": [round(w * 100, 2) for w in opt_weights.values()] if 'opt_weights' in locals() else [default_weight] * len(ticker_list)
        })

        col1, col2 = st.columns([2, 3])
        st.subheader("Custom Portfolio Weights")
        edited_weights_df = col1.data_editor(
            weights_df,
            num_rows="fixed",
            use_container_width=True,
            key="weights_editor",
            hide_index=True
        )

        # Normalize weights to sum to 100
        # Normalize weights to sum to 100 exactly (if not, auto-adjust last weight)
        total_weight = edited_weights_df["Custom Weight (%)"].sum()
        weights_adjusted = edited_weights_df.copy()
        if total_weight != 100 and total_weight != 0:
            st.warning(f"Total weight is {total_weight:.2f}%. Adjusting last weight to sum to 100%.")
            diff = 100 - total_weight
            # Adjust the last row
            last_idx = weights_adjusted.index[-1]
            weights_adjusted.at[last_idx, "Custom Weight (%)"] += diff
            total_weight = weights_adjusted["Custom Weight (%)"].sum()

        weights_dict = dict(zip(weights_adjusted["Ticker"], weights_adjusted["Custom Weight (%)"] / 100))
        # Now you can pass weights_dict to your portfolio_performance function

        if total_weight == 100:
            perf = portfolio_performance(
                data['prices'],
                weights_dict,
                benchmark=benchmark
            )
            st.write("Portfolio Performance", perf)

            # --- Plot custom portfolio trend ---
            # User input for starting balance
            start_balance = st.number_input(
                "Starting Balance ($)",
                min_value=1,
                value=1000,
                step=100,
                help="Set the starting balance for the hypothetical portfolio."
            )

            # Calculate daily returns for the selected tickers
            price_data = data['prices'][ticker_list]
            returns = price_data.pct_change().dropna()
            # Calculate portfolio returns using custom weights
            w = np.array([weights_dict[t] for t in ticker_list])
            port_ret = returns @ w
            # Calculate cumulative returns
            cumulative = (1 + port_ret).cumprod()
            portfolio_value = cumulative * start_balance
            # Plot using Plotly line chart for interactivity
            st.subheader("Custom Portfolio Value Over Time")
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='royalblue', width=3)
            ))
            fig_cum.update_layout(
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                title='Custom Portfolio Value Over Time',
                hovermode='x unified',
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_cum, use_container_width=True)


        if not data['prices'].empty and len(ticker_list) > 1:
            n_portfolios = 500
            results = []
            tickers = ticker_list

            for _ in range(n_portfolios):
                # Generate random weights that sum to 1
                w = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
                w_dict = dict(zip(tickers, w))
                metrics = portfolio_performance(data['prices'], w_dict, benchmark=benchmark)
                results.append({
                    'risk': metrics['annualized_risk'],
                    'return': metrics['annualized_return'],
                    'sharpe': metrics['sharpe_ratio'],
                    'weights': w_dict
                })

            ef_df = pd.DataFrame(results)

            # Add the custom portfolio as a separate point
            if total_weight == 100:
                user_point = {
                    'risk': perf['annualized_risk'],
                    'return': perf['annualized_return'],
                    'sharpe': perf['sharpe_ratio'],
                    'weights': weights_dict,
                    'Portfolio': 'Custom'
                }
                ef_df['Portfolio'] = 'Random'
                ef_df = pd.concat([ef_df, pd.DataFrame([user_point])], ignore_index=True)
            else:
                ef_df['Portfolio'] = 'Random'


            # Plotly interactive scatter plot
            fig = px.scatter(
                ef_df,
                x='risk',
                y='return',
                color='sharpe',
                color_continuous_scale='viridis',
                hover_data=['weights'],
                symbol='Portfolio',
                symbol_map={'Custom': 'star', 'Random': 'circle'},
                labels={
                    'risk': 'Annualized Risk (Std Dev)',
                    'return': 'Annualized Return',
                    'sharpe': 'Sharpe Ratio'
                },
                title='Efficient Frontier (Random Portfolios)'
            )

            # Make custom portfolio point larger and red, and add a circle indicator
            if total_weight == 100:
                # Highlight the custom portfolio with a star and a circle
                fig.update_traces(
                    selector=dict(mode='markers', marker_symbol='star'),
                    marker=dict(size=16, color='red', line=dict(width=2, color='black'))
                )
                # Add a circle indicator around the custom portfolio
                fig.add_shape(
                    type='circle',
                    xref='x', yref='y',
                    x0=perf['annualized_risk'] - 0.005, x1=perf['annualized_risk'] + 0.005,
                    y0=perf['annualized_return'] - 0.005, y1=perf['annualized_return'] + 0.005,
                    line=dict(color='red', width=3),
                    fillcolor='rgba(0,0,0,0)',
                    layer='above'
                )
            
            col2.plotly_chart(fig, use_container_width=True)