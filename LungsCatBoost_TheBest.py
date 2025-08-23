df = pd.read_csv("dataset.csv")
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO':0, 'YES':1})
x=df.drop(columns='LUNG_CANCER')
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)
info1 = []

lrs = list(np.array([1e-5]).repeat(9) * np.array(range(1, 10)))
lrs += list(np.array([1e-4]).repeat(9) * np.array(range(1, 10)))
lrs += list(np.array([1e-3]).repeat(9) * np.array(range(1, 10)))
lrs += list(np.array([1e-2]).repeat(9) * np.array(range(1, 10)))
lrs += list(np.array([1e-1]).repeat(5) * np.array(range(1, 6)))
lrs = list(np.round(lrs, 5))

for md in range(1, 6):
    for lr in lrs:
        cls = CatBoostClassifier(
            iterations=100000,
            max_depth=md,
            silent=True,
            learning_rate=lr,
            custom_metric=["Accuracy", "Recall", "Precision", "F1"],
            random_seed=123
        )
        cls.fit(
            x_train,
            y_train,
            cat_features=[
                'GENDER',
                'SMOKING',
                'YELLOW_FINGERS',
                'ANXIETY',
                'PEER_PRESSURE',
                'CHRONIC_DISEASE',
                'FATIGUE',
                'ALLERGY',
                'WHEEZING',
                'ALCOHOL_CONSUMING',
                'COUGHING',
                'SHORTNESS_OF_BREATH',
                'SWALLOWING_DIFFICULTY',
                'CHEST_PAIN'
            ],
            eval_set=(x_test, y_test),
            use_best_model=True,
            early_stopping_rounds=2000
        )

        y_pred = cls.predict(x_test)

        accuracy = cls.best_score_['validation']['Accuracy']
        precision = cls.best_score_['validation']['Precision']
        recall = cls.best_score_['validation']['Recall']
        f1 = cls.best_score_['validation']['F1']
        info1.append((accuracy, precision, recall, f1, md, lr))

        print(f"Best Model ({md}, {lr}): Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
