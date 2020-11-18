
#' Austrian Income and Living Conditions data (EU-SILC Austria (2011))

#' @source \url{https://ec.europa.eu/eurostat/cache/microdata/eusilc/AT_2011_EUSILC.zip}
#' @examples
#' \dontrun{
#'
#' fit <- lm(rent_ful ~ region+dwel_type+ tenure_status+leaking_roof+
#' bath_room+indoor_wc+dark_home+noisy_home+ bad_surrounding+
#' crime_surrounding+rooms,data = eusilc2011)
#' }
"eusilc2011"
