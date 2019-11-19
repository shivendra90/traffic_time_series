"""
Contains all Cython code.
"""

cpdef int show_bestScore(list train_set, list test_set):
    """
    Returns best cross-validated
    MAE and (p,d,q) order
    for a ts model.
    """
    cpdef list target = [values for values in train_set]
    cpdef list actuals = test_set
    cpdef list score = [10000, (0, 0, 0)]  # This will store the results
    cpdef int p
    cpdef int d
    cpdef int q
    for p in pList:
        for d in dList:
            for q in qList:
                order = (p, d, q)
                model = SARIMAX(target.astype("float32"), order=order)
                fit = model.fit(disp=False)
                preds = fit.forecast(len(actuals))
                error = mean_absolute_error(test.traffic_volume.astype("float32"), preds)
                if score[0] != 0 and error < score[0]:
                    score.pop(); score.pop()
                    score.append(error); score.append(order)

    best_score, best_order = score[0], score[1]
    return best_score, best_order
