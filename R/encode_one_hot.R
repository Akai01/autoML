#' One-hot encoding of chategorical or caracter variables.
#'
#' @description
#'
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#'
#' @importFrom fastDummies dummy_columns
#' @param data A data frame.
#' @param select_columns Vector of column names that you want to create dummy
#' variables from. If NULL (default), uses all character and factor columns.
#' @param remove_first_dummy Removes the first dummy of every variable such
#' that only n-1 dummies remain. This avoids multicollinearity issues in models.
#' @param remove_most_frequent_dummy Removes the most frequently observed
#' category such that only n-1 dummies remain. If there is a tie for most
#' frequent, will remove the first (by alphabetical order) category that is
#' tied for most frequent.
#' @param ignore_na If TRUE, ignores any NA values in the column.
#' If FALSE (default), then it will make a dummy column for value_NA
#' and give a 1 in any row which has a NA value.
#' @param split A string to split a column when multiple categories are in the
#'  cell. For example, if a variable is Pets and the rows are "cat",
#'   "dog", and "turtle", each of these pets would become its own dummy column.
#'    If one row is "cat, dog", then a split value of "," this row would have a
#'    value of 1 for both the cat and dog dummy columns.
#' @param remove_selected_columns If TRUE (not default), removes the columns
#' used to generate the dummy columns.
#' @examples
#' \dontrun{
#'
#' my_data <- encode_one_hot(my_data)
#' }
#' @author Resul Akay
#' @export

encode_one_hot <- function(data,
                           remove_selected_columns = TRUE,
                           select_columns = NULL,
                           remove_first_dummy = FALSE,
                           remove_most_frequent_dummy = FALSE,
                           ignore_na = FALSE,
                           split = NULL) {

  if(!"data.frame" %in% class(data)){
    stop("data must be a tidy data.frame")
  }


data <- fastDummies::dummy_columns(data, select_columns = select_columns,
                                  remove_first_dummy = remove_first_dummy,
                                  remove_most_frequent_dummy =
                                    remove_most_frequent_dummy,
                                  ignore_na = ignore_na,
                                  split = split,
                                  remove_selected_columns =
                                    remove_selected_columns)


  return(data)
}
