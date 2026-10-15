"""Постоянная палитра: территория, категории, статусы и прогноз."""
REGION = '#087F83'
NAVY = '#152D3D'
MUTED = '#536977'
GRID = '#E4EAF0'
COMPARISON = '#8A98A6'
FORECAST = '#6750A4'
STATUS = {'positive': '#178052', 'negative': '#B63B36',
          'attention': '#A66A10', 'neutral': '#536977'}
SEX = {'both': REGION, 'female': '#7860A5', 'male': '#3977A8'}
RESIDENCE = {'all': REGION, 'urban': '#3977A8', 'rural': '#A66A10'}
AGE = {'all': REGION, '0-14': '#417C97', '15-17': '#6489B0',
       '18-19': '#8A76AF', '20-39': '#5773A0', '40-59': '#96739C',
       '60+': '#93753E', '18+': '#5773A0'}
DIAGNOSIS = {'all': REGION, 'psychoses': '#5773A0',
             'nonpsychotic': '#8A76AF', 'intellectual': '#93753E'}


def chart_layout(figure, height=330):
    """Применить одни шрифты, сетку и фон без автопалитры Plotly."""
    figure.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Arial, sans-serif', color=NAVY),
        colorway=[REGION, COMPARISON, FORECAST],
        margin=dict(t=20, b=35, l=10, r=15),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        xaxis=dict(gridcolor=GRID, zeroline=False))
    return figure
