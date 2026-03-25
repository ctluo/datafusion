import pandas as pd

df_l = pd.read_excel('C:/Users/rochant/WorkBuddy/Claw/DataFusion/HSCM3_CN_Low-Fidelity.xlsx')
df_h = pd.read_excel('C:/Users/rochant/WorkBuddy/Claw/DataFusion/HSCM3_CN_High-Fidelity.xlsx')

print('LF shape:', df_l.shape)
print('LF cols:', list(df_l.columns))
print(df_l.head(5))
print()
print('HF shape:', df_h.shape)
print('HF cols:', list(df_h.columns))
print(df_h.head(5))
print()
print('LF describe:')
print(df_l[['H','Ma','sinA','CN']].describe())
print()
print('HF describe:')
print(df_h[['H','Ma','sinA','CN']].describe())
