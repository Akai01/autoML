#' Automatic model training: xgboost with linear booster.
#'
#' @description
#'
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#'
#' @param x A tidy data frame (samples are in rows and features are in columns)
#' containing features.
#' @param y A numeric or factor vector containing the outcome for each sample.
#' @param nrounds max number of boosting iterations. \code{0<= < Inf}
#' @param eta ontrol the learning rate: scale the contribution of each tree by
#' a factor of \code{0 < eta < 1} when it is added to the current approximation.
#' Used to prevent overfitting by making the boosting process more conservative.
#' Lower value for eta implies larger value for nrounds: low eta value means
#' model more robust to overfitting but slower to compute.
#' @param lambda L2 regularization term on weights.
#' @param alpha L1 regularization term on weights.
#' @param missing By default is set to NA, which means that NA values should
#' be considered as 'missing' by the algorithm. Sometimes, 0 or other extreme
#' value might be used to represent missing values. This parameter is only used
#' when input is a dense matrix.
#' @param weight A vector indicating the weight for each row of the input.
#' @param verbose If 0, xgboost will stay silent. If 1, it will print
#' information about performance. If 2, some additional information will be
#' printed out. Note that setting verbose > 0 automatically engages the
#' cb.print.evaluation(period=1) callback function
#' @param print_every_n Print each n-th iteration evaluation messages when
#' verbose>0. Default is 1 which means all messages are printed.
#' This parameter is passed to the cb.print.evaluation callback.
#' @param early_stopping_rounds If NULL, the early stopping function is not
#' triggered. If set to an integer k, training with a validation set will stop
#' if the performance doesn't improve for k rounds. Setting this parameter
#' engages the cb.early.stop callback.
#' @param maximize If feval and early_stopping_rounds are set, then this
#' parameter must be set as well. When it is TRUE, it means the larger the
#' evaluation score the better. This parameter is passed to the
#' cb.early.stop callback.
#' @param save_period When it is non-NULL, model is saved to disk after every
#' save_period rounds, 0 means save at the end. The saving is handled by the
#' cb.save.model callback.
#' @param xgb_model A previously built model to continue the training from.
#' Could be either an object of class xgb.Booster, or its raw data, or the name
#'  of a file with a previously saved model.
#' @param save_name The name or path for periodically saved model file.
#' @param callbacks a list of callback functions to perform various task during
#' boosting. See callbacks. Some of the callbacks are automatically created
#' depending on the parameters' values. User can provide either existing or
#' their own callback methods in order to customize the training process.
#' @param k An integer to specify the number of folds.
#' @param validation_error_metric Later
#' @param bo_iters Number of BO iterations.
#' @param init_design Length of the initial design
#' @param \dots Other parameters to pass to \code{params}.
#' @author Resul Akay
#' @details
#' For more details please check documentation of \code{\link[xgboost]{xgboost}}.
#'
#' For more information on parameters see
#' \url{https://sites.google.com/view/lauraepp/parameters}
#'
#'
#' @return
#' An object of class \code{xgb.Booster} with the following elements:
#' \itemize{
#'   \item \code{handle} a handle (pointer) to the xgboost model in memory.
#'   \item \code{raw} a cached memory dump of the xgboost model saved as R's
#'   \code{raw} type.
#'   \item \code{niter} number of boosting iterations.
#'   \item \code{evaluation_log} evaluation history stored as a
#'   \code{data.table} with the first column corresponding to iteration number
#'   and the rest corresponding to evaluation metrics' values.
#'   It is created by the \code{\link{cb.evaluation.log}} callback.
#'   \item \code{call} a function call.
#'   \item \code{params} parameters that were passed to the xgboost library.
#'   Note that it does not capture parameters changed by the
#'   \code{\link{cb.reset.parameters}} callback.
#'   \item \code{callbacks} callback functions that were either automatically assigned or
#'   explicitly passed.
#'   \item \code{best_iteration} iteration number with the best evaluation
#'   metric value (only available with early stopping).
#'   \item \code{best_ntreelimit} the \code{ntreelimit} value corresponding to
#'   the best iteration, which could further be used in \code{predict} method
#'   (only available with early stopping).
#'   \item \code{best_score} the best evaluation metric value during early
#'   stopping. (only available with early stopping).
#'   \item \code{feature_names} names of the training dataset features
#'   (only when column names were defined in training data).
#'   \item \code{nfeatures} number of features in training data.
#' }
#'
#' @examples
#' \dontrun{
#'
#' data("eusilc2011", package = "autoML")
#'
#' df <- split_data(eusilc2011)
#'
#' train_data <- df$train
#'
#' test_data <- df$test
#'
#' y <- train_data$rent_ful
#'
#' train_data$rent_ful <- NULL
#'
#' x <- train_data
#'
#' x <- encode_one_hot(x)
#'
#' fit_dart <- auto_xgblinear(x, y, nrounds = list(lower = 10, upper = 12),
#'                          k = 3, bo_iters = 4, init_design = 20)
#' }
#' @importFrom mlrMBO makeMBOControl setMBOControlTermination mbo
#' @importFrom xgboost xgboost
#' @importFrom smoof makeSingleObjectiveFunction
#' @importFrom lhs randomLHS
#' @importFrom ParamHelpers makeParamSet makeIntegerParam makeNumericParam
#' @importFrom ParamHelpers generateDesign getParamSet
#' @export
#'
#'
auto_xgblinear <- function(x,
                         y,
                         nrounds = list(lower = 10, upper = 100),
                         eta = list(lower = 0.001, upper = 0.6),
                         lambda = list(lower = 0, upper = 0.1),
                         alpha = list(lower = 0, upper = 0.1),
                         missing = NA,
                         weight = NULL,
                         verbose = 1,
                         print_every_n = 1L,
                         early_stopping_rounds = NULL,
                         maximize = NULL,
                         save_period = NULL,
                         save_name = "xgboost.model",
                         xgb_model = NULL,
                         callbacks = list(),
                         k = 5,
                         validation_error_metric = "RMSE",
                         bo_iters = 4, init_design = 15, ...){

  objective_function  <- smoof::makeSingleObjectiveFunction(
    name = "auto_xgblinear",
    fn =   function(param){
      acc <- xgboost_cv(x, y,
                        missing = missing,
                        weight = weight,
                        params = list(
                          booster = "gblinear",
                          eta = param["eta"],
                          lambda = param["lambda"],
                          alpha = param["alpha"]
                        ),
                        nrounds = param["nrounds"],
                        verbose = verbose,
                        print_every_n = print_every_n,
                        early_stopping_rounds = early_stopping_rounds,
                        maximize = maximize,
                        save_period = save_period,
                        save_name = save_name,
                        xgb_model = xgb_model,
                        callbacks = callbacks,
                        ..., k = k)

      a <- - mean(acc[,validation_error_metric])
      if(verbose>0){
        message(paste0("The mean error of ", k, " folds is ", - a))
      }
      return(a)
    },
    par.set = makeParamSet(
      makeIntegerParam("nrounds",
                       lower = nrounds[["lower"]],
                       upper = nrounds[["upper"]]),
      makeNumericParam("eta",
                       lower = eta[["lower"]],
                       upper = eta[["upper"]]),
      makeNumericParam("lambda",
                       lower = lambda[["lower"]],
                       upper = lambda[["upper"]]),
      makeNumericParam("alpha",
                       lower = alpha[["lower"]],
                       upper = alpha[["upper"]])
    ),
    minimize = FALSE
  )

  des <- ParamHelpers::generateDesign(
    n= init_design,
    par.set = ParamHelpers::getParamSet(objective_function),
    fun = lhs::randomLHS)

  control = makeMBOControl()
  control = setMBOControlTermination(control, iters = bo_iters)

  mlrmbo_result = mbo(fun = objective_function,
                      design = des,
                      control = control,
                      show.info = TRUE)
  params <- mlrmbo_result[["x"]]

  nrounds_bo <- params[["nrounds"]]

  params[["nrounds"]] <- NULL

  x <- as.matrix(x)
  dgsx <- xgboost::xgb.DMatrix(x, label = y)

  final_model <- xgboost::xgboost(data = dgsx,
                                  label = NULL,
                                  missing = missing,
                                  weight = weight,
                                  booster = "gblinear",
                                  params = params,
                                  nrounds = nrounds_bo,
                                  verbose = verbose,
                                  print_every_n = print_every_n,
                                  early_stopping_rounds = early_stopping_rounds,
                                  maximize = maximize,
                                  save_period = save_period,
                                  save_name = save_name,
                                  xgb_model = xgb_model,
                                  callbacks = callbacks,
                                  ...)
  ojct <- list("model" = final_model,"mlrmbo_result" = mlrmbo_result)

  class(ojct) <- "auto_xgb"

  return(ojct)
}
