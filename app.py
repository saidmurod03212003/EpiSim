import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from model import run_simulation, fit_parameters
from utils import calculate_advanced_metrics, generate_forecast, validate_data_columns

# --- Translations ---
LANGS = {
    "English": {
        "title": "🛡️ Advanced SEIRV Epidemic Simulator",
        "tab_sim": "📊 Simulation",
        "tab_comp": "🔄 Comparison",
        "tab_data": "🧬 Data Analysis",
        "sidebar_params": "Global Parameters",
        "pop": "Population (N)",
        "exposed": "Initially Exposed (E0)",
        "infected": "Initially Infected (I0)",
        "days": "Simulation Days",
        "variables": "Variables",
        "beta": "Infection Rate (β)",
        "sigma": "Incubation Rate (σ)",
        "gamma": "Recovery Rate (γ)",
        "nu": "Vaccination Rate (ν)",
        "forecast": "Show 30-Day Forecast",
        "r0": "R0 (Reproduction)",
        "peak": "Peak Infected",
        "total": "Total Infected",
        "duration": "Duration",
        "days_unit": "Days",
        "graph_title": "Disease Progression (SEIRV)",
        "export": "Export Data as CSV",
        "scenario_title": "Scenario Comparison",
        "sc_normal": "Base Case",
        "sc_lockdown": "Lockdown (50% reduction)",
        "sc_vacc": "Mass Vaccination (5x rate)",
        "data_fitting": "Real Data Fitting",
        "upload": "Upload Historical Data (CSV / Excel)",
        "csv_info": "Uploaded file must contain 'Day' and 'Infected' columns.",
        "fit_btn": "Calculate Best Fit",
        "success": "Data loaded successfully.",
        "best_params": "Best parameters found",
        "actual": "Actual Data",
        "fit_model": "Model Fit",
        "intro_1_title": "Why EpiSim? 🌍",
        "intro_1_text": "Epidemics spread fast, but our response can be faster. EpiSim is an advanced simulation tool designed to help you visualize the impact of diseases. Explore how lockdowns, vaccination strategies, and disease characteristics can change the course of an outbreak. It's a window into proactive public health!",
        "intro_2_title": "1. Global Parameters ⚙️",
        "intro_2_text": "Located on the sidebar, these define the starting point of our scenario.",
        "intro_2_pop": "**Population (N):** Total number of people.<br>**Initially Exposed (E0):** People who have the virus but are not yet infectious.<br>**Initially Infected (I0):** The first infectious cases.<br>**Simulation Days:** How long to run.",
        "intro_3_title": "2. Disease Variables 🦠",
        "intro_3_text": "These sliders control the behavior of the virus itself.",
        "intro_3_vars": "**Infection Rate (β):** How easily the disease spreads.<br>**Incubation Rate (σ):** How quickly exposed individuals become infectious.<br>**Recovery Rate (γ):** How fast infected people recover.<br>**Vaccination Rate (ν):** Proportion vaccinated daily.",
        "intro_4_title": "3. Real-time Analysis 📊",
        "intro_4_text": "As you change variables, the graph updates instantly!",
        "intro_4_metrics": "**R0 (Reproduction Number):** If > 1, the disease spreads.<br>**Peak Infected:** Maximum hospital capacity needed.<br>**Total Infected:** Overall impact of the epidemic.",
        "intro_5_title": "4. Advanced Tools 🔬",
        "intro_5_text": "Beyond basic simulation, EpiSim offers advanced modes.",
        "intro_5_tabs": "**Comparison Tab:** Compare different scenarios (e.g., Lockdowns vs. Mass Vaccination).<br>**Data Analysis Tab:** Upload real CSV data and let EpiSim find the best parameters automatically.",
        "btn_next": "Next ➡️",
        "btn_start": "Start Exploring 🚀",
        "btn_skip": "Skip Intro ⏭️",
        "trace_s": "Susceptible",
        "trace_e": "Exposed",
        "trace_i": "Infected",
        "trace_r": "Recovered",
        "trace_v": "Vaccinated",
        "trace_f": "Forecast",
        "human_export": "Human-readable export",
        "desc_day": "Simulation Day",
        "desc_s": "Number of people susceptible to the infection",
        "desc_e": "Number of people exposed but not yet infectious",
        "desc_i": "Number of infectious people",
        "desc_r": "Number of recovered people with immunity",
        "desc_v": "Number of vaccinated people",
        "desc_i_pct": "Percentage Infected",
        "desc_r_pct": "Percentage Recovered",
        "desc_v_pct": "Percentage Vaccinated",
        "calc": "Calculating...",
        "comp_intro_title": "Scenario Comparison Guide 🔄",
        "comp_intro_text": "This section compares different strategies simultaneously. See how a Lockdown (reducing infection rate) or Mass Vaccination (increasing vaccination rate) affects the epidemic curve compared to the Base Case.",
        "data_intro_title": "Data Analysis Guide 🧬",
        "data_intro_text": "Upload historical CSV data containing 'Day' and 'Infected' columns. The system will automatically calculate the best matching parameters for your real-world data."
    },
    "O'zbekcha": {
        "title": "🛡️ Ilg'or SEIRV Epidemik Simulyatori",
        "tab_sim": "📊 Simulyatsiya",
        "tab_comp": "🔄 Taqqoslash",
        "tab_data": "🧬 Ma'lumotlar tahlili",
        "sidebar_params": "Asosiy parametrlar",
        "pop": "Aholi soni (N)",
        "exposed": "Dastlabki ta'sirda bo'lganlar (E0)",
        "infected": "Dastlabki kasallanganlar (I0)",
        "days": "Simulyatsiya kunlari",
        "variables": "O'zgaruvchilar",
        "beta": "Yuqish darajasi (β)",
        "sigma": "Inkubatsiya davri (σ)",
        "gamma": "Tuzalish darajasi (γ)",
        "nu": "Emlash darajasi (ν)",
        "forecast": "30 kunlik bashoratni ko'rsatish",
        "r0": "R0 (Ko'payish soni)",
        "peak": "Eng yuqori kasallanish",
        "total": "Jami kasallanganlar",
        "duration": "Davomiyligi",
        "days_unit": "Kun",
        "graph_title": "Kasallikning rivojlanishi (SEIRV)",
        "export": "Ma'lumotlarni CSV sifatida yuklash",
        "scenario_title": "Ssenariylarni solishtirish",
        "sc_normal": "Asosiy holat",
        "sc_lockdown": "Lokatun (50% kamayish)",
        "sc_vacc": "Ommaviy emlash (5x tezlik)",
        "data_fitting": "Haqiqiy ma'lumotlarga moslash",
        "upload": "Tarixiy ma'lumotlarni yuklash (CSV / Excel)",
        "csv_info": "Yuklangan faylda 'Day' va 'Infected' ustunlari bo'lishi kerak.",
        "fit_btn": "Eng yaxshi moslikni hisoblash",
        "success": "Ma'lumotlar muvaffaqiyatli yuklandi.",
        "best_params": "Eng mos parametrlar topildi",
        "actual": "Haqiqiy ma'lumot",
        "fit_model": "Model natijasi",
        "intro_1_title": "Nega EpiSim? 🌍",
        "intro_1_text": "Epidemiyalar tez tarqaladi, ammo bizning javobimiz tezroq bo'lishi mumkin. EpiSim – bu kasallik ta'sirini vizuallashtirish uchun ilg'or simulyatsiya vositasi. Lokdaun, emlash kabi strategiyalar qanday ta'sir qilishini o'rganing. Bu shunchaki simulyator emas, sog'liqni saqlash kelajagiga nazar!",
        "intro_2_title": "1. Asosiy Parametrlar ⚙️",
        "intro_2_text": "Yon panelda joylashgan bu qiymatlar simulyatsiyaning boshlang'ich nuqtasini belgilaydi.",
        "intro_2_pop": "**Aholi soni (N):** Hududdagi jami odamlar soni.<br>**Dastlabki ta'sirda bo'lganlar (E0):** Virus yuqtirgan, lekin hali yuqumli bo'lmaganlar.<br>**Dastlabki kasallanganlar (I0):** Infeksiyani tarqatuvchi birinchi bemorlar.<br>**Simulyatsiya kunlari:** Jarayon qancha davom etishi.",
        "intro_3_title": "2. Kasallik O'zgaruvchilari 🦠",
        "intro_3_text": "Bu slayderlar orqali virusning xatti-harakatlarini boshqarasiz.",
        "intro_3_vars": "**Yuqish darajasi (β):** Kasallikning qanchalik oson o'tishi.<br>**Inkubatsiya davri (σ):** Ta'sirda bo'lganlarning yuqumli bo'lish tezligi.<br>**Tuzalish darajasi (γ):** Bemorlarning tuzalish tezligi.<br>**Emlash darajasi (ν):** Aholining kunlik emlanish foizi.",
        "intro_4_title": "3. Jonli Tahlil 📊",
        "intro_4_text": "Parametrlarni o'zgartirishingiz bilan grafik darhol yangilanadi!",
        "intro_4_metrics": "**R0 (Ko'payish soni):** Agar > 1 bo'lsa, kasallik tarqaladi.<br>**Eng yuqori kasallanish:** Shifoxonalarga tushadigan maksimal og'irlik.<br>**Jami kasallanganlar:** Epidemiyaning umumiy ta'siri.",
        "intro_5_title": "4. Ilg'or Vositalar 🔬",
        "intro_5_text": "Asosiy simulyatsiyadan tashqari qo'shimcha imkoniyatlar.",
        "intro_5_tabs": "**Taqqoslash:** Lokdaun yoki ommaviy emlash kabi ssenariylarni solishtirish.<br>**Ma'lumotlar tahlili:** Haqiqiy CSV ma'lumotlarni yuklab, unga mos keluvchi parametrlarni avtomatik topish.",
        "btn_next": "Keyingisi ➡️",
        "btn_start": "Boshlash 🚀",
        "btn_skip": "O'tkazib yuborish ⏭️",
        "trace_s": "Sog'lomlar",
        "trace_e": "Ta'sirda bo'lganlar",
        "trace_i": "Kasallanganlar",
        "trace_r": "Tuzalganlar",
        "trace_v": "Emlanganlar",
        "trace_f": "Bashorat",
        "human_export": "Tushunarli formatda yuklash",
        "desc_day": "Simulyatsiya kuni",
        "desc_s": "Hali infektsiyaga uchramagan odamlar soni",
        "desc_e": "Infeksiyaga duch kelgan, ammo hali yuqumli bo'lmagan shaxslar",
        "desc_i": "Infeksiyaga uchragan, yuqumli bo'lgan shaxslar soni",
        "desc_r": "Tuzalgan, immunitetga ega bo'lgan shaxslar soni",
        "desc_v": "Vaksinatsiyadan o'tgan shaxslar soni",
        "desc_i_pct": "Infeksiyalangan foiz",
        "desc_r_pct": "Tuzalgan foiz",
        "desc_v_pct": "Emlangan foiz",
        "calc": "Hisoblanmoqda...",
        "comp_intro_title": "Ssenariylarni Taqqoslash Yo'riqnomasi 🔄",
        "comp_intro_text": "Ushbu bo'lim turli strategiyalarni bir vaqtda taqqoslash imkonini beradi. Lokdaun (yuqishni kamaytirish) yoki Ommaviy emlash (emlashni tezlashtirish) kabi qarorlar asosiy holatga nisbatan qanday ijobiy ta'sir qilishini ko'ring.",
        "data_intro_title": "Ma'lumotlar Tahlili Yo'riqnomasi 🧬",
        "data_intro_text": "Tarixiy CSV ma'lumotlarni yuklang. Dastur sizning haqiqiy ma'lumotlaringizga eng mos keluvchi parametrlarni avtomatik ravishda hisoblab topib beradi."
    },
    "Русский": {
        "title": "🛡️ Продвинутый симулятор эпидемии SEIRV",
        "tab_sim": "📊 Симуляция",
        "tab_comp": "🔄 Сравнение",
        "tab_data": "🧬 Анализ данных",
        "sidebar_params": "Основные параметры",
        "pop": "Население (N)",
        "exposed": "Первично подверженные (E0)",
        "infected": "Первично инфицированные (I0)",
        "days": "Дни симуляции",
        "variables": "Переменные",
        "beta": "Скорость заражения (β)",
        "sigma": "Скорость инкубации (σ)",
        "gamma": "Скорость выздоровления (γ)",
        "nu": "Скорость вакцинации (ν)",
        "forecast": "Показать прогноз на 30 дней",
        "r0": "R0 (Репродукция)",
        "peak": "Пик инфицированных",
        "total": "Всего заразившихся",
        "duration": "Длительность",
        "days_unit": "Дней",
        "graph_title": "Прогрессия заболевания (SEIRV)",
        "export": "Экспорт в CSV",
        "scenario_title": "Сравнение сценариев",
        "sc_normal": "Базовый сценарий",
        "sc_lockdown": "Локдаун (снижение на 50%)",
        "sc_vacc": "Массовая вакцинация (5x скорость)",
        "data_fitting": "Подбор под реальные данные",
        "upload": "Загрузить данные (CSV / Excel)",
        "csv_info": "Загруженный файл должен содержать столбцы 'Day' и 'Infected'.",
        "fit_btn": "Рассчитать параметры",
        "success": "Данные успешно загружены.",
        "best_params": "Найдены оптимальные параметры",
        "actual": "Реальные данные",
        "fit_model": "Модель",
        "intro_1_title": "Зачем нужен EpiSim? 🌍",
        "intro_1_text": "Эпидемии распространяются быстро, но наша реакция может быть быстрее. EpiSim — это передовой инструмент для оценки влияния болезней. Изучите, как локдауны и вакцинация меняют ход вспышки. Это больше, чем симулятор!",
        "intro_2_title": "1. Основные параметры ⚙️",
        "intro_2_text": "Эти значения на боковой панели задают начальную точку.",
        "intro_2_pop": "**Население (N):** Общее число людей.<br>**Первично подверженные (E0):** Те, кто заражен, но еще не заразен.<br>**Первично инфицированные (I0):** Первые переносчики болезни.<br>**Дни симуляции:** Продолжительность процесса.",
        "intro_3_title": "2. Переменные болезни 🦠",
        "intro_3_text": "Эти ползунки управляют поведением вируса.",
        "intro_3_vars": "**Скорость заражения (β):** Насколько легко передается вирус.<br>**Скорость инкубации (σ):** Как быстро зараженные становятся заразными.<br>**Скорость выздоровления (γ):** Скорость поправки пациентов.<br>**Скорость вакцинации (ν):** Доля населения, вакцинируемая ежедневно.",
        "intro_4_title": "3. Анализ в реальном времени 📊",
        "intro_4_text": "При изменении переменных график обновляется мгновенно!",
        "intro_4_metrics": "**R0 (Репродукция):** Если > 1, болезнь распространяется.<br>**Пик инфицированных:** Максимальная нагрузка на больницы.<br>**Всего заразившихся:** Общий масштаб эпидемии.",
        "intro_5_title": "4. Продвинутые инструменты 🔬",
        "intro_5_text": "Дополнительные возможности EpiSim.",
        "intro_5_tabs": "**Сравнение:** Сравнение сценариев (локдаун, вакцинация).<br>**Анализ данных:** Загрузка реальных CSV-данных для автоматического подбора параметров.",
        "btn_next": "Далее ➡️",
        "btn_start": "Начать 🚀",
        "btn_skip": "Пропустить ⏭️",
        "trace_s": "Восприимчивые",
        "trace_e": "Контактные (в инкубации)",
        "trace_i": "Инфицированные",
        "trace_r": "Выздоровевшие",
        "trace_v": "Вакцинированные",
        "trace_f": "Прогноз",
        "human_export": "Человекочитаемый формат",
        "desc_day": "День симуляции",
        "desc_s": "Количество людей, восприимчивых к инфекции",
        "desc_e": "Количество людей в инкубационном периоде",
        "desc_i": "Количество зараженных и заразных людей",
        "desc_r": "Количество выздоровевших людей с иммунитетом",
        "desc_v": "Количество вакцинированных людей",
        "desc_i_pct": "Процент инфицированных",
        "desc_r_pct": "Процент выздоровевших",
        "desc_v_pct": "Процент вакцинированных",
        "calc": "Вычисление...",
        "comp_intro_title": "Руководство по сравнению сценариев 🔄",
        "comp_intro_text": "В этом разделе одновременно сравниваются различные стратегии. Посмотрите, как локдаун или массовая вакцинация влияют на кривую по сравнению с базовым сценарием.",
        "data_intro_title": "Руководство по анализу данных 🧬",
        "data_intro_text": "Загрузите исторические данные CSV. Система автоматически рассчитает наиболее подходящие параметры для ваших реальных данных."
    }
}

st.set_page_config(page_title="SEIRV Epidemic Simulator", layout="wide")

# Language Selector
def on_lang_change():
    st.session_state.intro_step = 1

lang_choice = st.sidebar.selectbox("🌐 Language / Til / Язык", list(LANGS.keys()), on_change=on_lang_change)
t_ = LANGS[lang_choice]

st.title(t_["title"])

# --- Onboarding Wizard ---
if "intro_step" not in st.session_state:
    st.session_state.intro_step = 1

step = st.session_state.intro_step

if step <= 5:
    css_target = ""
    if step == 2:
        css_target = '[data-testid="stSidebar"] { animation: pulseGlow 2s infinite; border: 2px solid #00f2fe; border-radius: 10px; }'
    elif step == 3:
        css_target = '[data-testid="stSlider"] { animation: pulseGlow 2s infinite; border: 2px solid #00f2fe; border-radius: 10px; padding: 5px; }'
    elif step == 4:
        css_target = '[data-testid="stMetric"], .js-plotly-plot { animation: pulseGlow 2s infinite; border: 2px solid #00f2fe; border-radius: 10px; }'
    elif step == 5:
        css_target = '[data-testid="stTabs"] { animation: pulseGlow 2s infinite; border: 2px solid #00f2fe; border-radius: 10px; }'

    st.markdown(f"""
        <style>
        @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.7); }}
            70% {{ box-shadow: 0 0 20px 15px rgba(0, 242, 254, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0, 242, 254, 0); }}
        }}
        {css_target}
        .wizard-card {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
            animation: slideDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        @keyframes slideDown {{
            from {{ opacity: 0; transform: translateY(-30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .wizard-title {{ font-size: 2rem; font-weight: bold; margin-bottom: 1rem; color: #00f2fe; }}
        .wizard-text {{ font-size: 1.1rem; line-height: 1.5; margin-bottom: 1rem; }}
        .wizard-highlight {{ background: rgba(255,255,255,0.1); padding: 1rem; border-left: 4px solid #00f2fe; border-radius: 8px; font-size: 1rem; margin-bottom: 1rem; }}
        </style>
    """, unsafe_allow_html=True)
    
    highlight_content = ""
    if step == 2: highlight_content = f'<div class="wizard-highlight">{t_["intro_2_pop"]}</div>'
    elif step == 3: highlight_content = f'<div class="wizard-highlight">{t_["intro_3_vars"]}</div>'
    elif step == 4: highlight_content = f'<div class="wizard-highlight">{t_["intro_4_metrics"]}</div>'
    elif step == 5: highlight_content = f'<div class="wizard-highlight">{t_["intro_5_tabs"]}</div>'

    st.markdown(f"""
        <div class="wizard-card">
            <div class="wizard-title">{t_[f"intro_{step}_title"]}</div>
            <div class="wizard-text">{t_[f"intro_{step}_text"]}</div>
            {highlight_content}
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col2:
        if st.button(t_['btn_skip'], use_container_width=True):
            st.session_state.intro_step = 6
            st.rerun()
    with col3:
        if st.button(t_['btn_next'] if step < 5 else t_['btn_start'], type="primary", use_container_width=True):
            st.session_state.intro_step += 1
            st.rerun()
    st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs([t_["tab_sim"], t_["tab_comp"], t_["tab_data"]])

# Sidebar Parameters
st.sidebar.markdown(f"### {t_['sidebar_params']}")
N = st.sidebar.number_input(t_["pop"], value=100000, step=1000)
E0 = st.sidebar.number_input(t_["exposed"], value=10)
I0 = st.sidebar.number_input(t_["infected"], value=1)
days = st.sidebar.slider(t_["days"], 30, 365, 180)

# --- Tab 1: Simulation ---
with tab1:
    col_params, col_viz = st.columns([1, 3])
    
    with col_params:
        st.subheader(t_["variables"])
        beta = st.slider(t_["beta"], 0.0, 2.0, 0.75)
        sigma = st.slider(t_["sigma"], 0.0, 1.0, 0.2)
        gamma = st.slider(t_["gamma"], 0.0, 1.0, 0.1)
        nu = st.slider(t_["nu"], 0.0, 0.1, 0.01, format="%.3f")
        forecast_on = st.checkbox(t_["forecast"])

    # Run core simulation
    t, (S, E, I, R, V) = run_simulation(N, E0, I0, 0, 0, beta, sigma, gamma, nu, days)
    r0, total_inf, duration = calculate_advanced_metrics(t, S, I, beta, gamma)

    with col_viz:
        # Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t_["r0"], f"{r0:.2f}")
        m2.metric(t_["peak"], f"{int(np.max(I)):,}")
        m3.metric(t_["total"], f"{int(total_inf):,}")
        m4.metric(t_["duration"], f"{int(duration)} {t_['days_unit']}")

        # Main Graph
        fig = go.Figure()
        colors = {'S': '#3498db', 'E': '#f1c40f', 'I': '#e74c3c', 'R': '#2ecc71', 'V': '#9b59b6'}
        fig.add_trace(go.Scatter(x=t, y=S, name=t_["trace_s"], line=dict(color=colors['S'])))
        fig.add_trace(go.Scatter(x=t, y=E, name=t_["trace_e"], line=dict(color=colors['E'])))
        fig.add_trace(go.Scatter(x=t, y=I, name=t_["trace_i"], line=dict(color=colors['I'], width=4)))
        fig.add_trace(go.Scatter(x=t, y=R, name=t_["trace_r"], line=dict(color=colors['R'])))
        fig.add_trace(go.Scatter(x=t, y=V, name=t_["trace_v"], line=dict(color=colors['V'])))

        if forecast_on:
            t_f, I_f = generate_forecast(t, I)
            fig.add_trace(go.Scatter(x=t_f, y=I_f, name=t_["trace_f"], line=dict(color='#e74c3c', dash='dash')))

        fig.update_layout(title=t_["graph_title"], template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
        
        # Download — clean, readable export (integers + %, downsampled to ~50 rows)
        # Option for human‑readable formatting
        human_readable = st.checkbox(t_["human_export"], value=True, key="export_fmt")
        step = max(1, len(t) // 50)
        base_data = {
            "Day":          t[::step].astype(int),
            "Susceptible":  S[::step].astype(int),
            "Exposed":      E[::step].astype(int),
            "Infected":     I[::step].astype(int),
            "Recovered":    R[::step].astype(int),
            "Vaccinated":   V[::step].astype(int),
            "Infected_%":   (I[::step] / N * 100).round(2),
            "Recovered_%":  (R[::step] / N * 100).round(2),
            "Vaccinated_%": (V[::step] / N * 100).round(2),
        }
        if human_readable:
            # Add formatted strings with thousands separators for large numbers
            fmt_data = {
                "Susceptible_fmt":  list(map(lambda x: f"{x:,}", base_data["Susceptible"])),
                "Exposed_fmt":      list(map(lambda x: f"{x:,}", base_data["Exposed"])) ,
                "Infected_fmt":     list(map(lambda x: f"{x:,}", base_data["Infected"])) ,
                "Recovered_fmt":    list(map(lambda x: f"{x:,}", base_data["Recovered"])) ,
                "Vaccinated_fmt":   list(map(lambda x: f"{x:,}", base_data["Vaccinated"])) ,
            }
            df_export = pd.DataFrame({
                "Day": base_data["Day"],
                "Susceptible": fmt_data["Susceptible_fmt"],
                "Exposed": fmt_data["Exposed_fmt"],
                "Infected": fmt_data["Infected_fmt"],
                "Recovered": fmt_data["Recovered_fmt"],
                "Vaccinated": fmt_data["Vaccinated_fmt"],
                "Infected_%": base_data["Infected_%"],
                "Recovered_%": base_data["Recovered_%"],
                "Vaccinated_%": base_data["Vaccinated_%"],
            })
        else:
            df_export = pd.DataFrame(base_data)

        # Translated column descriptions (second row)
        col_desc = {
            "Day": t_["desc_day"],
            "Susceptible": t_["desc_s"],
            "Exposed": t_["desc_e"],
            "Infected": t_["desc_i"],
            "Recovered": t_["desc_r"],
            "Vaccinated": t_["desc_v"],
            "Infected_%": t_["desc_i_pct"],
            "Recovered_%": t_["desc_r_pct"],
            "Vaccinated_%": t_["desc_v_pct"]
        }
        # Insert description as first row
        desc_row = pd.DataFrame([col_desc])
        df_export = pd.concat([desc_row, df_export], ignore_index=True)

        st.download_button(t_["export"], df_export.to_csv(index=False), "seirv_results.csv", "text/csv")

# --- Tab 2: Comparison ---
with tab2:
    st.subheader(t_["scenario_title"])
    st.markdown(f"""
        <div style="background: rgba(0, 242, 254, 0.05); border-left: 4px solid #00f2fe; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <strong style="color: #00f2fe; font-size: 1.1rem;">{t_['comp_intro_title']}</strong><br>
            <span style="color: #e0e0e0;">{t_['comp_intro_text']}</span>
        </div>
    """, unsafe_allow_html=True)
    scenarios = {
        t_["sc_normal"]: {"beta": beta, "nu": nu},
        t_["sc_lockdown"]: {"beta": beta * 0.5, "nu": nu},
        t_["sc_vacc"]: {"beta": beta, "nu": nu * 5}
    }
    
    fig_comp = go.Figure()
    for name, p in scenarios.items():
        _, (_, _, I_s, _, _) = run_simulation(N, E0, I0, 0, 0, p['beta'], sigma, gamma, p['nu'], days)
        fig_comp.add_trace(go.Scatter(x=t, y=I_s, name=name))
    
    fig_comp.update_layout(template="plotly_white")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- Tab 3: Data Analysis ---
with tab3:
    st.subheader(t_["data_fitting"])
    st.markdown(f"""
        <div style="background: rgba(0, 242, 254, 0.05); border-left: 4px solid #00f2fe; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <strong style="color: #00f2fe; font-size: 1.1rem;">{t_['data_intro_title']}</strong><br>
            <span style="color: #e0e0e0;">{t_['data_intro_text']}</span>
        </div>
    """, unsafe_allow_html=True)
    file = st.file_uploader(t_["upload"], type=["csv", "xlsx", "xls"])
    if file:
        try:
            if file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file)
            else:
                data = pd.read_csv(file)
            
            if validate_data_columns(data):
                st.success(t_["success"])
                if st.button(t_["fit_btn"]):
                    with st.spinner(t_["calc"]):
                        b_fit, g_fit = fit_parameters(data['Day'].values, data['Infected'].values, N, E0, I0, 0, 0, sigma, nu)
                        st.write(f"**{t_['best_params']}:** β={b_fit:.4f}, γ={g_fit:.4f}")
                        t_f, (_, _, I_f, _, _) = run_simulation(N, E0, I0, 0, 0, b_fit, sigma, g_fit, nu, int(data['Day'].max()))
                        
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Scatter(x=data['Day'], y=data['Infected'], mode='markers', name=t_["actual"]))
                        fig_f.add_trace(go.Scatter(x=t_f, y=I_f, name=t_["fit_model"], line=dict(color='red')))
                        st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.error(t_["csv_info"])
        except Exception as e:
            st.error(f"Faylni o'qishda xatolik yuz berdi: {e}" if lang_choice == "O'zbekcha" else f"Error reading file: {e}" if lang_choice == "English" else f"Ошибка чтения файла: {e}")
