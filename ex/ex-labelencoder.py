from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(map(lambda x: x.lower(), ["paris", "paris", "tokyo", "amsterdam", "Paris"]))

print("classes", list(le.classes_))

print("transto integer", le.transform(["tokyo", "tokyo", "paris"]))

print("reverse transform", list(le.inverse_transform([2, 2, 1])))
