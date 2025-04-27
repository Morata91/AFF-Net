import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 読み込むファイル名を指定
result_file = 'saves/evaluation/all.log'  # ←ここにログファイルのパスを書く

# データを保存するための辞書
avg = []
person = -1
with open(result_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        a = line.split()
        if a[0] == 'Test':
            person += 1
            avg.append([])
        else:
            avg[person].append(a[-1])


print(avg)
print(avg[11][:])

exit()
# 各被験者ごとに
# - 12番目の誤差
# - 最小誤差
label_12_errors = {}
min_errors = {}

for person, labels in data.items():
    if 12 in labels:
        label_12_errors[person] = labels[12]
    min_errors[person] = min(labels.values())

# 12エポック目の誤差の平均を計算
if label_12_errors:
    mean_epoch12_error = np.mean(list(label_12_errors.values()))
else:
    mean_epoch12_error = None  # もし12エポックのデータがなかったらNone

# グラフ作成
persons = sorted(label_12_errors.keys())
label12_values = [label_12_errors[p] for p in persons]
min_values = [min_errors[p] for p in persons]

x = np.arange(len(persons))
width = 0.35

plt.figure(figsize=(14,7))
plt.bar(x - width/2, label12_values, width, label='12th Epoch Error')
plt.bar(x + width/2, min_values, width, label='Minimum Error')
plt.xlabel('Person')
plt.ylabel('Error')
plt.title('Comparison of 12th Epoch Error and Minimum Error per Person')
plt.xticks(x, persons, rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs('saves/analytics', exist_ok=True)
plt.savefig('saves/analytics/epoch12_vs_min_error.png')
plt.show()

# テキスト出力（一覧表示）
print("\n=== Summary ===")
for person in persons:
    epoch12 = label_12_errors.get(person, None)
    min_error = min_errors.get(person, None)
    print(f"{person}: 12 Epoch Error = {epoch12:.5f}, Minimum Error = {min_error:.5f}")

# 12エポック目の誤差の平均を表示
if mean_epoch12_error is not None:
    print(f"\nAverage Error at 12th Epoch: {mean_epoch12_error:.5f}")
else:
    print("\nNo data available for 12th Epoch error.")
print("================\n")