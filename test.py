import numpy as np
import analysis

dataset_path = 'models/data.csv'
model_name = 'fraud_model'
features = ['userId','profileCompleted','visitedCountries','depositAmount','totalSpendingsInt','spendings','maxDailySpending']
labels = []

test_data = [191670,1,2,2492270,14,82939,20000]
target_data = np.array(test_data).reshape(-1, len(test_data))

x = analysis.predict(target_data, features, labels, dataset_path, model_name, True)
y = x["prediction"]
print(y)

z = x["pred"]