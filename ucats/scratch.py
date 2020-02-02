from imfun.cluster import som
def second_stage_svd_old(collection, fsh, do_spatial=False, n_clusters=20, wsize=64, stride=32):

    all_svd_signals = vstack([c[0] for c in collection])
    #     squares =  list(map(tuple, ucats.make_grid(fsh, wsize,stride)))
    #     counts = np.zeros()
    #     for sqx in squares:
    #         signals = vstack([c[0] for c in collection if in_square(c[3],sqx)])
    #         clust = skcluster.KMeans(n_clusters)
    #         labels = claust.fit_predict(signals)
    #         sqx_approx = np.zeros(all_svd_signals.shape)
    #         for i in range(n_clusters):
    #             group = labels==i
    #             u,s,vh = svd(all_svd_signals[group],False)
    #             r = ucats.min_ncomp(s, (u.shape[0],vh.shape[1]))+1
    #             #w = weight_components(all_svd_signals[group],vh,r)
    #             approx = (u[:,:r])@diag(s[:r])@vh[:r]
    #             sqx_approx[group] += approx
    #         pass

    all_centers = array([patch_center(c[3]) for c in collection])
    C = _pairwise_euclidean_distances(all_centers)


    print('Clustering %d temporal components into %d clusters'%(len(all_svd_signals),n_clusters))
    #clust = skcluster.AgglomerativeClustering(n_clusters,affinity='euclidean',linkage='ward',connectivity=C)
    #clust = skcluster.KMeans(n_clusters)
    clust = skcluster.MiniBatchKMeans(n_clusters,batch_size=2*n_clusters)
    labels = clust.fit_predict(all_svd_signals)
    #labels = som(all_svd_signals,(n_clusters,1))
    all_svd_approx = np.zeros(all_svd_signals.shape)

    #print('Doing 2nd-stage SVD within clusters', flush=True)
    for i in tqdm(unique(labels), desc='Doing 2nd-stage SVD in clusters'):
        group = labels == i
        u,s,vh = svd(all_svd_signals[group],False)
        r = ucats.min_ncomp(s, (u.shape[0],vh.shape[1]))+1
        #w = weight_components(all_svd_signals[group],vh,r)
        approx = (u[:,:r])@diag(s[:r])@vh[:r]
        all_svd_approx[group] = approx


    grouped_signals = []
    kstart = 0
    for c in collection:
        l = len(c[0])
        grouped_signals.append(all_svd_approx[kstart:kstart+l])
        kstart += l

    spatial_filter_sizes = [c[1].shape[1] for c in collection]
    lmax = np.max(spatial_filter_sizes)


    if do_spatial:
        print('Processing spatial components', flush=True)
        all_svd_filters = vstack([c[1] for c in collection if c[1].shape[1]==lmax])
        all_svd_spatial_approx = np.zeros(all_svd_filters.shape)

        #clust = skcluster.AgglomerativeClustering(n_clusters, affinity='l1',linkage='average')
        clust = skcluster.KMeans(n_clusters,n_jobs=1)
        labels = clust.fit_predict(all_svd_filters)
        for i in tqdm(unique(labels)):
            group = labels==i
            if not any(group):
                continue
            u,s,vh = svd(all_svd_filters[group],False)
            r = ucats.min_ncomp(s, (u.shape[0],vh.shape[1]))+1
            approx = u[:,:r]@diag(s[:r])@vh[:r]
            all_svd_spatial_approx[group] = approx

        kstart = 0
        grouped_filters = []
        for c in collection:
            if c[1].shape[1] == lmax:
                l = len(c[1])
                grouped_filters.append(all_svd_spatial_approx[kstart:kstart+l])
                kstart += l
            else:
                grouped_filters.append(c[1])
    else:
        grouped_filters = [c[1] for c in collection]

    out_coll = [(tsc, ssc) + c[2:]
                for c,tsc,ssc in zip(collection, grouped_signals, grouped_filters)]
    return out_coll
