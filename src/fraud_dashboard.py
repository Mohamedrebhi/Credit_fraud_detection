import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# Initialize session state for storing real-time data
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(
        columns=['timestamp', 'amount', 'fraud_probability', 'status']
    )

def add_transaction(new_data):
    """Add new transaction to the session state"""
    st.session_state.transactions = pd.concat([
        st.session_state.transactions,
        pd.DataFrame([new_data])
    ]).tail(100)  # Keep last 100 transactions

def create_metrics(df):
    st.subheader("Real-time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    current_time = pd.Timestamp.now()
    
    with col1:
        st.metric(
            "Transactions (Last Hour)", 
            len(df[df['timestamp'] > current_time - pd.Timedelta(hours=1)]),
            delta=len(df[df['timestamp'] > current_time - pd.Timedelta(minutes=5)]),
            delta_color="normal"
        )
    with col2:
        high_risk = len(df[df['fraud_probability'] > 0.7])
        prev_high_risk = len(df[
            (df['fraud_probability'] > 0.7) & 
            (df['timestamp'] > current_time - pd.Timedelta(hours=2)) &
            (df['timestamp'] <= current_time - pd.Timedelta(hours=1))
        ])
        st.metric("High Risk Alerts", high_risk, 
                 delta=high_risk - prev_high_risk,
                 delta_color="inverse")
    with col3:
        avg_prob = df['fraud_probability'].mean() if not df.empty else 0
        prev_avg = df[df['timestamp'] > current_time - pd.Timedelta(hours=1)]['fraud_probability'].mean() if not df.empty else 0
        st.metric("Average Risk Score", 
                 f"{avg_prob:.2%}",
                 delta=f"{(avg_prob - prev_avg):.2%}",
                 delta_color="inverse")
    with col4:
        total_amount = df['amount'].sum() if not df.empty else 0
        high_risk_amount = df[df['fraud_probability'] > 0.7]['amount'].sum() if not df.empty else 0
        st.metric("High Risk Amount", 
                 f"${high_risk_amount:,.2f}",
                 delta=f"{(high_risk_amount/total_amount*100):.1f}% of total",
                 delta_color="inverse")

def plot_real_time_trends(df):
    if not df.empty:
        st.subheader("Live Fraud Risk Trends")
        
        # Sort and resample data for smoother visualization
        df_sorted = df.sort_values('timestamp').set_index('timestamp')
        df_resampled = df_sorted.resample('1min').agg({
            'fraud_probability': 'mean',
            'amount': 'sum',
            'status': lambda x: (x == 'high_risk').mean()
        }).fillna(method='ffill')
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add fraud probability line
        fig.add_trace(
            go.Scatter(
                x=df_resampled.index,
                y=df_resampled['fraud_probability'],
                name='Fraud Risk',
                line=dict(color='blue', width=2),
                fill='tozeroy'
            )
        )
        
        # Add amount bars
        fig.add_trace(
            go.Bar(
                x=df_resampled.index,
                y=df_resampled['amount'],
                name='Transaction Amount',
                yaxis='y2',
                opacity=0.3,
                marker_color=np.where(df_resampled['status'] > 0.5, 'red', 'green')
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Real-time Fraud Risk Monitor',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Time',
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(
                title='Fraud Probability',
                tickformat='.1%',
                range=[0, 1],
                side='left',
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='Amount ($)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add threshold line
        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk Threshold (70%)",
            annotation_position="bottom right"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Real-time Fraud Detection Dashboard")
    
    # Transaction Analysis Form
    with st.sidebar:
        st.subheader("Transaction Analysis")
        time_now = int(datetime.now().timestamp() % 86400)
        time = st.number_input("Time (seconds)", value=time_now, min_value=0, max_value=86400)
        amount = st.number_input("Amount ($)", value=100.0, min_value=0.0)
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        
        if st.button("Analyze Transaction"):
            features = [time, amount, v1, v2]
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"features": features},
                    timeout=5  # Add timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add new transaction to session state
                    new_transaction = {
                        'timestamp': pd.Timestamp.now(),
                        'amount': amount,
                        'fraud_probability': result['fraud_probability'],
                        'status': 'high_risk' if result['fraud_probability'] > 0.7 else 'normal'
                    }
                    add_transaction(new_transaction)
                    
                    # Show result
                    if result['fraud_probability'] > 0.7:
                        st.sidebar.error(f"⚠️ High Risk - {result['fraud_probability']:.2%}")
                    else:
                        st.sidebar.success(f"✅ Low Risk - {result['fraud_probability']:.2%}")
                else:
                    st.sidebar.error(f"API Error: Status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Error: Cannot connect to the API. Make sure the API server is running.")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        create_metrics(st.session_state.transactions)
    
    with col2:
        if not st.session_state.transactions.empty:
            plot_real_time_trends(st.session_state.transactions)
    
    # Recent Transactions Table
    if not st.session_state.transactions.empty:
        st.subheader("Recent Transactions")
        
        # Get the most recent transactions
        recent_transactions = st.session_state.transactions.sort_values('timestamp', ascending=False).head(10)
        
        # Apply styling for high-risk transactions
        def highlight_high_risk(row):
            return ['background-color: #ffcccc' if row.fraud_probability > 0.7 else '' for _ in row]
        
        styled_df = recent_transactions.style.apply(highlight_high_risk, axis=1)
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            use_container_width=True
        )
    else:
        st.info("No transactions analyzed yet. Use the sidebar form to analyze transactions.")

if __name__ == "__main__":
    main()