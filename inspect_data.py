import pandas as pd

df_l = pd.read_excel('C:/Users/rochant/WorkBuddy/Claw/DataFusion/LTV_Low-Fidelity.xlsx')
df_h = pd.read_excel('C:/Users/rochant/WorkBuddy/Claw/DataFusion/LTV_High-Fidelity.xlsx')

print('=== LF Data ===')
print('Shape:', df_l.shape)
print('Columns:', list(df_l.columns))
print(df_l.head(5).to_string())
print()
print('=== HF Data ===')
print('Shape:', df_h.shape)
print('Columns:', list(df_h.columns))
print(df_h.head(5).to_string())
print()
print('LF Mach unique:', sorted(df_l['Mach'].unique()))
print('HF Mach unique:', sorted(df_h['Mach'].unique()))
print('LF CN stats:', df_l['CN'].describe())
print('HF CN stats:', df_h['CN'].describe())
