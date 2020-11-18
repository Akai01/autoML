if(require(testthat)){

  test_that("tests for some arguments in auto_xgbtree", {

    library(mlrMBO)

    data("eusilc2011", package = "autoML")

    df <- split_data(eusilc2011)

    train_data <- df$train

    test_data <- df$test

    y <- train_data$rent_ful

    train_data$rent_ful <- NULL

    x <- train_data

    x <- encode_one_hot(x)

    fit_tree <- auto_xgbtree(x, y, nrounds = list(lower = 10, upper = 12),
                             k = 3, bo_iters = 4, init_design = 20)

    fitted <- predict(fit_tree, x)

    a <- class(fitted)

    expect_that(a, equals("numeric"))

  })
}
