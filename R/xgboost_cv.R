#' @importFrom xgboost xgboost xgb.DMatrix
#' @importFrom caret createFolds
#' @importFrom plyr aaply splat
xgboost_cv <- function(x, y,
                        missing = NA,
                        weight = NULL,
                        params = list(),
                        nrounds,
                        verbose = 1,
                        print_every_n = 1L,
                        early_stopping_rounds = NULL,
                        maximize = NULL,
                        save_period = NULL,
                        save_name = "xgboost.model",
                        xgb_model = NULL,
                        callbacks = list(),
                        ..., k = 10){

  folds <- caret::createFolds(1:NROW(y), k = k, list = T, returnTrain = T)
  fold_list <- data.frame("i"=1:k)

  acc <- plyr::aaply(fold_list, 1, plyr::splat(function(i) {
    train_x <- data.frame(x[folds[[i]], ])
    validation_x <- data.frame(x[-folds[[i]], ])
    train_y <- y[folds[[i]]]
    validation_y <- y[-folds[[i]]]

    if(verbose>0){
      message(paste("Training model for fold ", i))
    }

    train_x <- as.matrix(train_x)
    dtrain <- xgboost::xgb.DMatrix(train_x, label = train_y)

    fit <-    xgboost::xgboost(data = dtrain,
                               label = NULL,
                               missing = missing,
                               weight = weight,
                               params = params,
                               nrounds = nrounds,
                               verbose = verbose,
                               print_every_n = print_every_n,
                               early_stopping_rounds = early_stopping_rounds,
                               maximize = maximize,
                               save_period = save_period,
                               save_name = save_name,
                               xgb_model = xgb_model,
                               callbacks = callbacks,
                               ...)

    validation_x <- as.matrix(validation_x)

    validation_x <- xgboost::xgb.DMatrix(validation_x)

    newpred <- predict(fit, validation_x)
    acc <- accuracy(pred =  newpred, actual =  validation_y)
    acc

  }), .expand = TRUE,
  .progress = "none",
  .inform = FALSE,
  .drop = TRUE,
  .parallel = FALSE,
  .paropts = NULL
  )
  return(acc)
}
