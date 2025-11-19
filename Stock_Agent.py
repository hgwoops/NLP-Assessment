import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="Qwen 全能投研终端 Pro",
    page_icon="📈",
    layout="wide"
)

# CSS 样式增强
st.markdown("""
    <style>
    /* 优化 Metric 卡片样式 */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .stMetric {
        background-color: #161920;
        border: 1px solid #30333d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* 调整 Spinner 颜色 */
    .stSpinner > div {
        border-top-color: #00C9FF !important;
    }
    /* 按钮样式优化 */
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Qwen-Plus 全能投研终端 (Pro版)")

# ==========================================
# 2. 侧边栏：全局控制 (增加 Form 表单)
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统设置")
    api_key = st.text_input("阿里云 DashScope API Key", type="password")
    
    st.markdown("---")
    
    # 使用 Form 表单，避免滑块拖动时频繁刷新页面
    with st.form(key='settings_form'):
        st.header("🎯 标的与参数")
        symbol = st.text_input("股票代码", value="NVDA", help="美股输代码，A股如 600519.SS")
        
        st.subheader("🛠️ 回测配置")
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.number_input("短期均线", min_value=3, max_value=100, value=20)
        with col2:
            long_window = st.number_input("长期均线", min_value=10, max_value=300, value=50)
            
        initial_capital = st.number_input("初始资金", value=100000, step=10000)
        
        # 新增：交易成本设置
        trans_cost = st.slider("单边交易费率 (%)", 0.0, 0.5, 0.05, step=0.01) / 100
        
        # 提交按钮
        submit_button = st.form_submit_button(label="🚀 应用参数 & 运行回测")
        
    st.markdown("### 🧠 AI 实验室")
    if st.button("✨ AI 自动寻找最佳参数"):
        st.session_state['do_optimize'] = True

# ==========================================
# 3. 核心逻辑函数 (增加缓存与增强逻辑)
# ==========================================

@st.cache_resource
def get_agent(api_key):
    """创建带有缓存的 Agent 实例"""
    if not api_key: return None
    return Agent(
        name="Full Stack Analyst",
        model=OpenAIChat(
            id="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        ),
        # 确保开启 Markdown 和工具调用
        tools=[DuckDuckGoTools(), YFinanceTools(stock_price=True, company_news=True, stock_fundamentals=True)],
        instructions=[
            "你是一位拥有 20 年经验的华尔街对冲基金经理，擅长宏观分析与量化交易。",
            "【思维链要求】在回答用户问题之前，请遵循以下思考步骤：",
            "1. **观察 (Observe)**: 检查用户提供的或工具获取的数据（价格趋势、估值指标、新闻）。",
            "2. **分析 (Analyze)**: 结合基本面（低估/高估）和技术面（金叉/死叉/超买超卖）进行交叉验证。",
            "3. **结论 (Conclude)**: 给出明确的观点（看多/看空/观望），并提示核心风险点。",
            "【格式要求】",
            "- 使用 Markdown 表格对比数据。",
            "- 关键结论请加粗。",
            "- 如果缺乏数据，请明确告知'数据不足'，不要编造。",
            "- 语气专业、客观、冷静。"
        ],
        markdown=True,
        show_tool_calls=True 
    )

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="5y"):
    try:
        # 增加 auto_adjust=True 有时能解决数据对齐问题
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df.empty: 
            print(f"【调试信息】获取到的数据为空: {symbol}") # 会打印在运行 Streamlit 的黑色终端里
            return None
        return df
    except Exception as e:
        print(f"【调试信息】报错详情: {e}") # 关键：看这里打印了什么！
        st.error(f"内部报错: {e}") # 将报错直接显示在网页上供你查看
        return None

@st.cache_data(ttl=3600)
def get_fundamental_info(symbol):
    """获取基本面数据并缓存"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        financials = ticker.financials
        cashflow = ticker.cashflow
        return info, financials, cashflow
    except Exception:
        return {}, pd.DataFrame(), pd.DataFrame()
    
def generate_context_summary(symbol, info, metrics, params):
    """
    将复杂的 DataFrame 和字典数据转换为自然语言摘要，
    供 Agent 在回答问题时参考。
    """
    # 1. 基本面摘要
    fund_str = "暂无基本面数据"
    if info and isinstance(info, dict):
        fund_str = f"""
        - 市值: {info.get('marketCap', 'N/A')}
        - 市盈率 (PE): {info.get('trailingPE', 'N/A')}
        - ROE: {info.get('returnOnEquity', 'N/A')}
        - 行业: {info.get('sector', '未知')}
        """

    # 2. 策略回测摘要
    # 修复点：先检查 metrics 是否为字典 (dict)，防止报错
    if isinstance(metrics, dict):
        bt_str = f"""
        - 策略参数: 短均线={params[0]}, 长均线={params[1]}
        - 总收益率: {metrics.get('Total Return', 0):.2%}
        - 年化收益 (CAGR): {metrics.get('CAGR', 0):.2%}
        - 最大回撤: {metrics.get('Max Drawdown', 0):.2%}
        - 夏普比率: {metrics.get('Sharpe Ratio', 0):.2f}
        - 交易次数: {metrics.get('Trade Count', 0)}
        """
    else:
        # 如果 metrics 是字符串（错误信息）或 None
        bt_str = f"回测数据不可用 (原因: {metrics if metrics else '数据不足'})"
    
    context = f"""
    【当前全维度数据快照 - 标的: {symbol}】
    
    [1. 财务基本面]
    {fund_str}
    
    [2. 技术面双均线策略表现]
    {bt_str}
    
    [指令]
    请基于以上数据回答用户问题。如果数据不可用，请告知用户可能需要检查股票代码或市场数据。
    """
    return context

def run_backtest_optimized(df_origin, short_w, long_w, capital, cost_rate):
    """
    执行增强版双均线策略回测
    包含：交易成本、夏普比率、年化收益
    """
    try:
        if df_origin is None or df_origin.empty: return None, "无数据"
        
        df = df_origin.copy()
        
        # 1. 计算指标
        df['SMA_Short'] = df['Close'].rolling(window=short_w).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_w).mean()
        
        # 2. 生成信号 (1: 持有, 0: 空仓)
        # 昨天收盘的均线决定今天的开盘操作，所以信号需要 shift(1)
        # 逻辑：若 Close > SMA，则 Next Day 持有
        df['Signal_Raw'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal_Raw'] = 1
        
        # 实际持仓：昨天的信号决定今天
        df['Position'] = df['Signal_Raw'].shift(1)
        df['Position'] = df['Position'].fillna(0)
        
        # 3. 计算基础收益
        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy_Raw_Return'] = df['Position'] * df['Daily_Return']
        
        # 4. 计算交易成本
        # Position 发生变化（0->1 买入, 1->0 卖出）时产生费用
        df['Trade_Action'] = df['Position'].diff().abs() # 1 表示有交易
        df['Cost'] = df['Trade_Action'] * cost_rate
        
        # 5. 净收益 = 策略收益 - 成本
        df['Strategy_Net_Return'] = df['Strategy_Raw_Return'] - df['Cost']
        
        # 6. 累计净值
        df['Cum_Bench_Return'] = (1 + df['Daily_Return']).cumprod() * capital
        df['Cum_Strategy_Return'] = (1 + df['Strategy_Net_Return']).cumprod() * capital
        
        # 7. 计算高级指标
        final_equity = df['Cum_Strategy_Return'].iloc[-1]
        total_return = (final_equity / capital) - 1
        
        # 年化收益 (CAGR)
        days = (df.index[-1] - df.index[0]).days
        cagr = (final_equity / capital) ** (365 / days) - 1 if days > 0 else 0
        
        # 最大回撤
        cum_max = df['Cum_Strategy_Return'].cummax()
        drawdown = (df['Cum_Strategy_Return'] - cum_max) / cum_max
        max_drawdown = drawdown.min()
        
        # 夏普比率 (假设无风险利率为 3%)
        rf = 0.03
        excess_returns = df['Strategy_Net_Return'] - (rf / 252)
        std_dev = excess_returns.std() * np.sqrt(252)
        sharpe_ratio = (excess_returns.mean() * 252) / std_dev if std_dev != 0 else 0
        
        metrics = {
            "Total Return": total_return,
            "CAGR": cagr,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Final Capital": final_equity,
            "Trade Count": df['Trade_Action'].sum()
        }
        return df, metrics
    except Exception as e:
        return None, str(e)
    
def optimize_strategy(df, capital, cost_rate):
    """
    简单的网格搜索，寻找夏普比率最高的均线组合
    """
    best_sharpe = -100
    best_params = (0, 0)
    best_metrics = {}
    
    # 搜索空间：短期 5-50，长期 20-200 (步长加大以节省计算时间)
    # 注意：Streamlit 中计算太久会超时，这里做简化处理
    short_range = range(5, 60, 10) 
    long_range = range(20, 150, 20)
    
    progress_bar = st.progress(0)
    total_steps = len(short_range) * len(long_range)
    step = 0
    
    for s in short_range:
        for l in long_range:
            step += 1
            progress_bar.progress(step / total_steps)
            
            if s >= l: continue # 短期必须小于长期
            
            # 复用之前的回测逻辑，但我们需要提取其中的计算核心，
            # 为了代码简洁，这里直接调用 run_backtest_optimized 
            # (生产环境建议拆分计算逻辑以提速)
            _, metrics = run_backtest_optimized(df, s, l, capital, cost_rate)
            
            if metrics and metrics['Sharpe Ratio'] > best_sharpe:
                best_sharpe = metrics['Sharpe Ratio']
                best_params = (s, l)
                best_metrics = metrics
                
    progress_bar.empty()
    return best_params, best_metrics

# ==========================================
# 4. 主界面布局 (Tabs)
# ==========================================

tab1, tab2, tab3 = st.tabs(["💬 智能对话 & 实时看盘", "🏢 深度基本面分析", "🔄 策略回测系统"])

# --- TAB 1: 智能对话 & 实时看盘 ---
with tab1:
    col_chart, col_chat = st.columns([3, 1])
    
    with col_chart:
        st.subheader(f"📈 {symbol} 实时走势")
              
        # 2. 获取数据
        with st.spinner("正在加载全量数据..."):
            df_price = get_stock_data(symbol, period="max")
            
        if df_price is not None and not df_price.empty:
            fig = go.Figure()
            
            # K线
            fig.add_trace(go.Candlestick(
                x=df_price.index,
                open=df_price['Open'], high=df_price['High'],
                low=df_price['Low'], close=df_price['Close'],
                name='股价',
                increasing_line_color='#ff4b4b', 
                decreasing_line_color='#00c853' 
            ))
            
            # 均线
            sma_short = df_price['Close'].rolling(window=short_window).mean()
            sma_long = df_price['Close'].rolling(window=long_window).mean()
            
            fig.add_trace(go.Scatter(x=df_price.index, y=sma_short, line=dict(color='#00C9FF', width=1.5), name=f'SMA {short_window}'))
            fig.add_trace(go.Scatter(x=df_price.index, y=sma_long, line=dict(color='#FFD700', width=1.5), name=f'SMA {long_window}'))

            # =================================================
            # 📐 最终版视图逻辑
            # =================================================
            
            ipo_date = df_price.index[0]
            last_date = df_price.index[-1]
            
            # 初始视图：最近 6 个月
            initial_start = last_date - pd.DateOffset(months=6)
            if initial_start < ipo_date: initial_start = ipo_date

            fig.update_layout(
                height=700,
                template="plotly_dark",
                hovermode="x unified",
                
                # --- X 轴 ---
                xaxis=dict(
                    # 左右边界限制，防止拖到无数据区域
                    minallowed=ipo_date - pd.Timedelta(days=1),
                    maxallowed=last_date + pd.Timedelta(days=1),
                    
                    # 初始聚焦在最近 6 个月
                    range=[initial_start, last_date + pd.Timedelta(hours=6)],
                    
                    rangeslider=dict(visible=False),
                    rangebreaks=[dict(bounds=["sat", "mon"])],
                    
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1月", step="month", stepmode="backward"),
                            dict(count=3, label="3月", step="month", stepmode="backward"),
                            dict(count=6, label="6月", step="month", stepmode="backward"),
                            dict(count=1, label="1年", step="year", stepmode="backward"),
                            dict(count=3, label="3年", step="year", stepmode="backward"),
                            dict(step="all", label="Max")
                        ]),
                        bgcolor="#262730", font=dict(color="white", size=12), activecolor="#00C9FF"
                    )
                ),
                
                # --- Y 轴 ---
                yaxis=dict(
                    title="价格",
                    tickprefix="$",

                    autorange=True, 
                    
                    # 2. 核心设置：防止出现从 0 开始的大片空白
                    # 'normal' 表示：根据数据范围自动决定起点，不强制包含 0
                    rangemode="normal", 
                    
                    fixedrange=False,
                    type='linear'
                ),
                
                legend=dict(orientation="h", y=1.02, x=0),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("无法加载数据")

    with col_chat:
        st.subheader("🤖 AI 投研顾问")
        
        # --- A. 处理自动寻优逻辑 ---
        if st.session_state.get('do_optimize', False):
            with st.spinner("AI 正在疯狂回测历史数据，寻找最优解..."):
                # ... (获取数据和寻优代码保持不变) ...
                df_opt = get_stock_data(symbol, period="5y")
                if df_opt is not None:
                    best_p, best_m = optimize_strategy(df_opt, initial_capital, trans_cost)
                    
                    # --- 修改开始：使用 st.toast 跨 Tab 通知 ---
                    msg = f"✅ 寻优完成！最佳参数: {best_p[0]} / {best_p[1]} (夏普: {best_m['Sharpe Ratio']:.2f})"
                    st.toast(msg, icon="🎉") 
                    # --- 修改结束 ---
                    
                    st.success(f"寻优完成！最佳均线: 短期 {best_p[0]} / 长期 {best_p[1]}")
                    st.info(f"该组合夏普比率: {best_m['Sharpe Ratio']:.2f}, 总回报: {best_m['Total Return']:.2%}")
                    st.caption("请在左侧侧边栏手动更新上述参数以查看详细图表。")
            st.session_state['do_optimize'] = False

        # --- B. 准备 AI 的上下文数据 (Real-time Context) ---
        # 1. 获取当前的基本面
        curr_info, _, _ = get_fundamental_info(symbol)
        # 2. 获取当前的回测结果 (即使在 Tab 3 没点开，这里也要算一下给 AI 看)
        df_context = get_stock_data(symbol, period="5y")

        # 增加一层保护：如果连 df_context 都没有，直接给空字典
        if df_context is not None and not df_context.empty:
            _, curr_metrics = run_backtest_optimized(df_context, short_window, long_window, initial_capital, trans_cost)
        else:
            curr_metrics = "无法获取历史股价数据，请检查股票代码"
        # 3. 生成系统级 Context 字符串
        system_context = generate_context_summary(symbol, curr_info, curr_metrics, (short_window, long_window))

        # --- C. 聊天界面 ---
        chat_container = st.container(height=600)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(f"问我关于 {symbol} 的任何问题..."):
            if not api_key:
                st.error("请先设置 API Key")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                chat_container.chat_message("user").write(prompt)

                with chat_container.chat_message("assistant"):
                    agent = get_agent(api_key)
                    if agent:
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        # --- 核心修改：将 Context 拼接到用户 Prompt 前面 ---
                        # 这样 AI 就能基于当前的图表和回测数据回答，而不是瞎编
                        augmented_prompt = f"{system_context}\n\n用户问题: {prompt}"
                        
                        try:
                            # 注意：我们发给 Agent 的是 augmented_prompt，但界面上用户只看到自己的 prompt
                            resp_stream = agent.run(augmented_prompt, stream=True)
                            for chunk in resp_stream:
                                content = ""
                                if hasattr(chunk, "content"): content = chunk.content
                                elif isinstance(chunk, str): content = chunk
                                if content:
                                    full_response += content
                                    response_placeholder.markdown(full_response + "▌")
                            response_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            st.error(f"AI 响应错误: {e}")

# --- TAB 2: 深度基本面分析 ---
with tab2:
    st.header(f"🏢 {symbol} 基本面深度透视")
    
    # 使用缓存获取基本面数据
    info, financials, cashflow = get_fundamental_info(symbol)

    if info:
        # 辅助函数
        def fmt_num(n):
            if not isinstance(n, (int, float)): return "N/A"
            if abs(n) > 1e12: return f"{n/1e12:.2f}T"
            if abs(n) > 1e9: return f"{n/1e9:.2f}B"
            if abs(n) > 1e6: return f"{n/1e6:.2f}M"
            return f"{n:.2f}"
            
        def fmt_pct(n): return f"{n*100:.2f}%" if isinstance(n, (int, float)) else "N/A"

        st.subheader("1. 核心财务指标矩阵")
        
        # Row 1
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("总市值", fmt_num(info.get("marketCap")))
        with c2: st.metric("市盈率 (TTM)", fmt_num(info.get("trailingPE")))
        with c3: st.metric("Forward PE", fmt_num(info.get("forwardPE")))
        with c4: st.metric("PEG Ratio", fmt_num(info.get("pegRatio")), help="< 1 通常表示低估")

        # Row 2
        c5, c6, c7, c8 = st.columns(4)
        with c5: st.metric("ROE", fmt_pct(info.get("returnOnEquity")))
        with c6: st.metric("毛利率", fmt_pct(info.get("grossMargins")))
        with c7: st.metric("净利率", fmt_pct(info.get("profitMargins")))
        with c8: st.metric("营收增长 (YoY)", fmt_pct(info.get("revenueGrowth")))

        st.markdown("---")

        st.subheader("2. 财务趋势可视化")
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.caption("📊 营收 vs 净利润")
            if not financials.empty:
                fin_T = financials.T.sort_index()
                fin_T.index = pd.to_datetime(fin_T.index).year
                
                fig_inc = go.Figure()
                if 'Total Revenue' in fin_T.columns:
                    fig_inc.add_trace(go.Bar(x=fin_T.index, y=fin_T['Total Revenue'], name='营收', marker_color='#2E86C1'))
                
                net_col = next((c for c in ['Net Income', 'Net Income Common Stockholders'] if c in fin_T.columns), None)
                if net_col:
                    fig_inc.add_trace(go.Bar(x=fin_T.index, y=fin_T[net_col], name='净利润', marker_color='#F4D03F'))
                
                fig_inc.update_layout(barmode='group', height=350, template="plotly_dark", margin=dict(t=10, b=10))
                st.plotly_chart(fig_inc, use_container_width=True)

        with chart_c2:
            st.caption("💰 现金流结构")
            if not cashflow.empty:
                cf_T = cashflow.T.sort_index()
                cf_T.index = pd.to_datetime(cf_T.index).year
                
                fig_cf = go.Figure()
                op_col = next((c for c in ['Operating Cash Flow', 'Total Cash From Operating Activities'] if c in cf_T.columns), None)
                if op_col:
                    fig_cf.add_trace(go.Bar(x=cf_T.index, y=cf_T[op_col], name='经营现金流', marker_color='#2ECC71'))
                
                if 'Free Cash Flow' in cf_T.columns:
                    fig_cf.add_trace(go.Bar(x=cf_T.index, y=cf_T['Free Cash Flow'], name='自由现金流', marker_color='#E74C3C'))
                
                fig_cf.update_layout(barmode='group', height=350, template="plotly_dark", margin=dict(t=10, b=10))
                st.plotly_chart(fig_cf, use_container_width=True)
        # ... (接在 现金流结构 图表代码之后) ...

        st.markdown("---")
        st.subheader("3. 📰 AI 舆情情感分析")
        
        col_news_btn, col_news_display = st.columns([1, 3])
        
        with col_news_btn:
            st.caption("AI 将检索最新新闻并计算市场情绪得分 (-1 为极度悲观，+1 为极度乐观)")
            analyze_news = st.button("🔍 扫描新闻 & 分析情绪")
            
        if analyze_news:
            with col_news_display:
                if not api_key:
                    st.error("请先设置 API Key")
                else:
                    agent = get_agent(api_key)
                    with st.spinner(f"正在阅读 {symbol} 的全网新闻..."):
                        try:
                            # 1. 构造专门的 Prompt 让 Agent 读书
                            news_prompt = f"""
                            请使用工具搜索关于 {symbol} (股票代码) 最近 7 天的 5 条重要财经新闻。
                            
                            任务要求：
                            1. 简要列出这 5 条新闻的标题。
                            2. 综合分析这些新闻对股价的影响。
                            3. 给出一个“市场情绪得分”（范围 -1.0 到 +1.0，0 为中性）。
                            4. 输出格式：
                               - **新闻摘要**: ...
                               - **情绪得分**: `+0.x` 或 `-0.x`
                               - **关键理由**: ...
                            """
                            response = agent.run(news_prompt)
                            st.markdown(response.content)
                            
                        except Exception as e:
                            st.error(f"分析失败: {str(e)}")
        
        # AI 研报部分保持原逻辑，但利用缓存数据
        st.markdown("---")
        st.subheader("4. 🤖 AI 深度点评")
        if st.button("生成深度研报", type="primary"):
            if not api_key:
                st.error("请设置 API Key")
            else:
                analysis_prompt = f"""
                请分析 {symbol} (行业: {info.get('sector', '未知')})。
                数据: PE={info.get('trailingPE')}, ROE={info.get('returnOnEquity')}, 
                毛利率={info.get('grossMargins')}, 债务权益比={info.get('debtToEquity')}。
                请给出：1.估值评价 2.风险提示 3.投资建议。
                """
                agent = get_agent(api_key)
                with st.spinner("正在生成研报..."):
                    try:
                        st.markdown(agent.run(analysis_prompt).content)
                    except Exception as e:
                        st.error(str(e))
    else:
        st.warning("未找到基本面数据，可能是 ETF 或数据源暂缺。")

# --- TAB 3: 策略回测 (逻辑增强版) ---
with tab3:
    st.header(f"🔄 策略沙箱: 双均线趋势跟踪")
    
    # 校验参数
    if short_window >= long_window:
        st.error("⚠️ 错误: 短期均线必须小于长期均线。请在左侧侧边栏调整并点击应用。")
    else:
        # 使用缓存的历史数据运行回测
        df_base = get_stock_data(symbol, period="5y") # 回测通常需要更长数据
        df_bt, res = run_backtest_optimized(df_base, short_window, long_window, initial_capital, trans_cost)
        
        if df_bt is not None:
            # 1. 核心指标 (增加 Sharpe 和 CAGR)
            k1, k2, k3, k4 = st.columns(4)
            
            total_ret = res['Total Return']
            color = "normal" if total_ret >= 0 else "inverse"
            
            k1.metric("总收益率", f"{total_ret*100:.2f}%", delta_color=color)
            k2.metric("年化收益 (CAGR)", f"{res['CAGR']*100:.2f}%")
            k3.metric("最大回撤", f"{res['Max Drawdown']*100:.2f}%", delta_color="inverse")
            k4.metric("夏普比率 (Sharpe)", f"{res['Sharpe Ratio']:.2f}", help=">1 为佳，>2 非常优秀")
            
            st.markdown(f"**期末资产:** ${res['Final Capital']:,.2f} | **交易次数:** {int(res['Trade Count'])} | **单边费率:** {trans_cost*100}%")
         
            st.markdown("---")
            
            # 新增：AI 策略诊断
            with st.expander("🩺 AI 策略诊断报告 (点击展开)", expanded=True):
                if not api_key:
                    st.warning("请输入 API Key 以获取 AI 对该回测结果的专业评价")
                else:
                    # 自动触发评价（或者做成按钮）
                    if st.button("🧠 生成回测诊断"):
                        with st.spinner("AI 正在分析你的策略漏洞..."):
                            diag_agent = get_agent(api_key)
                            
                            # 这里的 prompt 专门针对量化陷阱
                            diag_prompt = f"""
                            【策略诊断任务】
                            用户在 {symbol} 上测试了双均线策略 (Short={short_window}, Long={long_window})。
                            
                            回测结果:
                            - 总回报: {res['Total Return']:.2%}
                            - 夏普比率: {res['Sharpe Ratio']:.2f}
                            - 最大回撤: {res['Max Drawdown']:.2%}
                            - 交易次数: {res['Trade Count']}
                            
                            请像一位严格的风控官一样指出问题：
                            1. **过拟合风险**: 交易次数是否太少？（少于 10 次通常统计意义不大）
                            2. **收益风险比**: 夏普比率是否大于 1？如果小于 1，这个策略是否值得执行？
                            3. **回撤承受力**: 最大回撤是否会导致爆仓？
                            4. **改进建议**: 如何调整均线参数可能更好？
                            """
                            
                            try:
                                resp = diag_agent.run(diag_prompt)
                                st.markdown(resp.content)
                            except Exception as e:
                                st.error(str(e))
            
            st.markdown("---")

            # 2. 资金曲线
            st.subheader("💸 策略净值 vs 基准")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cum_Bench_Return'], name='基准 (Buy & Hold)', line=dict(dash='dash', color='gray')))
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cum_Strategy_Return'], name='策略净值 (费后)', line=dict(color='#AB00FF', width=2), fill='tonexty', fillcolor='rgba(171, 0, 255, 0.1)'))
            
            fig_bt.update_layout(height=400, template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0))
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # 3. 信号复盘
            st.subheader("🔎 买卖点位复盘")
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Candlestick(x=df_bt.index, open=df_bt['Open'], high=df_bt['High'], low=df_bt['Low'], close=df_bt['Close'], name='股价'))
            
            # 绘制均线
            fig_sig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['SMA_Short'], name=f'SMA {short_window}', line=dict(width=1, color='yellow')))
            fig_sig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['SMA_Long'], name=f'SMA {long_window}', line=dict(width=1, color='blue')))

            # 标记信号
            buy_sigs = df_bt[df_bt['Trade_Action'] == 1]
            # 筛选买入动作 (Position 变为 1)
            real_buys = df_bt[(df_bt['Position'] == 1) & (df_bt['Position'].shift(1) == 0)]
            real_sells = df_bt[(df_bt['Position'] == 0) & (df_bt['Position'].shift(1) == 1)]

            fig_sig.add_trace(go.Scatter(
                x=real_buys.index, y=df_bt.loc[real_buys.index, 'Low']*0.98,
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00FF00'), name='买入'
            ))
            
            fig_sig.add_trace(go.Scatter(
                x=real_sells.index, y=df_bt.loc[real_sells.index, 'High']*1.02,
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='#FF0055'), name='卖出'
            ))

            fig_sig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_sig, use_container_width=True)
