class Clustering:
    def __init__(self, data, score_type="silhouette", pca_plot=False, type="kmeans"):

        self.data = data.dropna(axis=1)

        self.pca_plot = pca_plot
        self.type = type
        self.score_type = score_type
        self.n_components = self.get_n_components()

        print(f"no. of components: {self.n_components}")
        self.preprocessor = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA())])
        self.preprocessor.fit(self.data)
        self.preprocessed_data = self.preprocessor.transform(self.data)

        if self.type == "kmeans":
            self.kmeans_kwargs = {"init": "random", "n_init": 50, "max_iter": 500, "random_state": 22, }
            self.n_clusters = self.get_n_clusters()
            self.kmeans = KMeans(n_clusters=self.n_clusters, **self.kmeans_kwargs)
            self.kmeans.fit(self.preprocessed_data)
            self.labels = self.kmeans.labels_

        elif self.type == "spectral":
            self.spectral_kwargs = {"n_init": 50, "random_state": 22, "affinity": 'nearest_neighbors', }
            # 'eigen_solver':"arpack",
            self.n_clusters = self.get_n_clusters()
            self.spectral = SpectralClustering(n_clusters=self.n_clusters, **self.spectral_kwargs)
            self.spectral.fit(self.preprocessed_data)
            self.labels = self.spectral.labels_

        elif self.type == "heirarchical":
            self.heirarchical_kwargs = {"metric": 'euclidean', "linkage": 'ward'}
            self.n_clusters = self.get_n_clusters()
            self.heirarchical = AgglomerativeClustering(n_clusters=self.n_clusters, **self.heirarchical_kwargs)
            self.heirarchical.fit(self.preprocessed_data)
            self.labels = self.heirarchical.labels_

        print(f"no. of clusters: {self.n_clusters}")

    def plot_groups(self):
        fte_colors = {
            -1: "#003428",
            0: "#008fd5",
            1: "#fc4f30",
            2: "#000000",
            3: "#ffffff",
            4: "#389241",
            5: "#434822"}

        if self.type == "kmeans":
            a = self.kmeans.fit_predict(self.preprocessed_data)
            fig, ax = plt.subplots()
            sns.scatterplot(x=self.preprocessed_data[:, 0], y=self.preprocessed_data[:, 1], hue=a, ax=ax)
            kmeans_silhouette = silhouette_score(self.preprocessed_data, self.kmeans.labels_).round(2)
            ax.set(title=f"{self.type} Clustering:    Silhouette: {kmeans_silhouette}")

        elif self.type == "spectral":
            a = self.spectral.fit_predict(self.preprocessed_data)
            fig, ax = plt.subplots()
            sns.scatterplot(x=self.preprocessed_data[:, 0], y=self.preprocessed_data[:, 1], hue=a, ax=ax)
            spectral_silhouette = silhouette_score(self.preprocessed_data, self.spectral.labels_).round(2)
            ax.set(title=f"{self.type} Clustering:    Silhouette: {spectral_silhouette}")

        elif self.type == "heirarchical":
            a = self.heirarchical.fit_predict(self.preprocessed_data)
            fig, ax = plt.subplots()
            sns.scatterplot(x=self.preprocessed_data[:, 0], y=self.preprocessed_data[:, 1], hue=a, ax=ax)
            heirarchical_silhouette = silhouette_score(self.preprocessed_data, self.heirarchical.labels_).round(2)
            ax.set(title=f"{self.type} Clustering:    Silhouette: {heirarchical_silhouette}")

        else:
            print("Invalid Input!")

    def get_n_components(self):
        pca = PCA(random_state=22)

        x_pca = pca.fit_transform(self.data)

        exp_var_pca = pca.explained_variance_ratio_

        cum_sum_eigenvalues = np.cumsum(exp_var_pca)

        n = -1

        for i in range(len(cum_sum_eigenvalues)):
            if cum_sum_eigenvalues[i] > 0.90:
                n = i
                break

        if n == -1:
            for i in range(len(cum_sum_eigenvalues)):
                if cum_sum_eigenvalues[i] > 0.85:
                    n = i
                    break

        if self.pca_plot:
            plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
                    label='Individual explained variance')
            plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
                     label='Cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal component index')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        return n

    def get_n_clusters(self):

        coeff = []

        if self.score_type == "silhouette":
            if self.type == "kmeans":
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
                    kmeans.fit(self.preprocessed_data)
                    score = silhouette_score(self.preprocessed_data, kmeans.labels_)
                    coeff.append(score)
            elif self.type == "spectral":
                for k in range(2, 11):
                    spectral = SpectralClustering(n_clusters=k, **self.spectral_kwargs)
                    spectral.fit(self.preprocessed_data)
                    score = silhouette_score(self.preprocessed_data, spectral.labels_)
                    coeff.append(score)
            elif self.type == "heirarchical":
                for k in range(2, 11):
                    heirarchical = AgglomerativeClustering(n_clusters=k, **self.heirarchical_kwargs)
                    heirarchical.fit(self.preprocessed_data)
                    score = silhouette_score(self.preprocessed_data, heirarchical.labels_)
                    coeff.append(score)

            plt.style.use("fivethirtyeight")
            plt.plot(range(2, 11), coeff)
            plt.xticks(range(2, 11))
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.show()

        elif self.score_type == "sse":
            if self.type == "kmeans":
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
                    kmeans.fit(self.preprocessed_data)
                    coeff.append(kmeans.inertia_)
            elif self.type == "spectral":
                for k in range(2, 11):
                    spectral = SpectralClustering(n_clusters=k, **self.spectral_kwargs)
                    spectral.fit(self.preprocessed_data)
                    coeff.append(spectral.inertia_)
            elif self.type == "heirarchical":
                for k in range(2, 11):
                    heirarchical = AgglomerativeClustering(n_clusters=k, **self.heirarchical_kwargs)
                    heirarchical.fit(self.preprocessed_data)
                    coeff.append(heirarchical.inertia_)

            plt.style.use("fivethirtyeight")
            plt.plot(range(2, 11), coeff)
            plt.xticks(range(2, 11))
            plt.xlabel("Number of Clusters")
            plt.ylabel("SSE")
            plt.show()

        kl = KneeLocator(range(2, 11), coeff, curve="convex", direction="decreasing")
        print(kl.elbow)

        return kl.elbow





