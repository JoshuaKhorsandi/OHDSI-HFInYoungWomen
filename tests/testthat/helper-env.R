writeEnvFile <- function(lines, env = parent.frame()) {
  keys <- c("DB_DBMS", "DB_SERVER", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_CONNECTION_STRING",
            "DB_EXTRA_SETTINGS", "DB_PATH_TO_DRIVER", "CDM_DATABASE_SCHEMA",
            "COHORT_DATABASE_SCHEMA", "COHORT_TABLE", "COHORT_ID", "OUTPUT_FOLDER",
            "PYTHON_VIRTUALENV", "JAVA_PARAMETERS", "ANDROMEDA_TEMP_FOLDER")
  Sys.unsetenv(keys)
  oldOptions <- options(java.parameters = getOption("java.parameters"))
  path <- tempfile(fileext = ".env")
  writeLines(lines, path)
  withr::defer({
    Sys.unsetenv(keys)
    options(oldOptions)
    unlink(path)
  }, envir = env)
  path
}
