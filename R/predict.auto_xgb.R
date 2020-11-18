#' Predict an auto_xgboost object.
#' @description
#'
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#'
#' @param object an auto_xgboost object.
#' @param newdata newdata
#' @param \dots Other arguments
#' @author Resul Akay
#' @importFrom xgboost xgb.DMatrix
#' @importFrom stats predict
#' @export
predict.auto_xgb <- function(object, newdata, ...) {

  model <- object[["model"]]

  newdata <- as.matrix(newdata)
  newdata <- xgboost::xgb.DMatrix(newdata)

  pred <- stats::predict(model, newdata, ...)

  return(pred)
}
