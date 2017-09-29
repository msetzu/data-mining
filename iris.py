import pandas

iris = pandas.read_csv('iris.csv')
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

classes = set(iris['class'])
sepal_length = iris['sepal_length']
sepal_width = iris['sepal_width']
petal_length = iris['petal_length']
petal_width = iris['petal_width']

# min&max fields per attribute
min_sepal_length = min(set(sepal_length))
max_sepal_length = max(set(sepal_length))
min_sepal_width = min(set(sepal_width))
max_sepal_width = max(set(sepal_width))
min_petal_length = min(set(petal_length))
max_petal_length = max(set(petal_length))
min_petal_length = min(set(petal_width))
max_petal_length = max(set(petal_width))

# standard deviations
sepal_length_std = sepal_length.std()
sepal_width_std = sepal_width.std()
petal_length_std = petal_length.std()
petal_width_std = petal_width.std()

# correlation coefficients
mean_sepal_length = sepal_length.mean()
mean_sepal_width = sepal_width.mean()
mean_petal_length = petal_length.mean()
mean_petal_width = petal_width.mean()

sepal_pearsons_coefficient = sum([e*f for e in list(map(lambda l: l-mean_sepal_length,sepal_length)) for f in list(map(lambda l: l-mean_sepal_width,sepal_width))])
petal_pearsons_coefficient = sum([e*f for e in list(map(lambda l: l-mean_petal_length,petal_length)) for f in list(map(lambda l: l-mean_petal_width,petal_width))])


print('Classes: ' + str(classes))
print('Minimum sepal length: ' + str(min_sepal_length))
print('Maximum sepal length: ' + str(max_sepal_length))
print('Minimum petal length: ' + str(min_petal_length))
print('Maximum petal length: ' + str(max_petal_length))
print('Sepal length std: ' + str(sepal_length_std))
print('Sepal width std: ' + str(sepal_width_std))
print('Petal length std: ' + str(petal_length_std))
print('Petal width std: ' + str(petal_width_std))