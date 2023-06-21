import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
#plt.imshow(china) do this if u want axis

china.shape

#scaling and reshaping the data
data = china / 255.0
data = data.reshape(427 * 640, 3)
data.shape

#mini batch k means 
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(3)
kmeans.fit(data)

new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

china_recolored = new_colors.reshape(china.shape)
fig, ax = plt.subplots(1, 2, figsize = (16,6), subplot_kw = dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace = 0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size = 16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size = 16)
