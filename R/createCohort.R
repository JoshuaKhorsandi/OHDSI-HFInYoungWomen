#' Create a cohort using a db connection
#'
#' @param connectionDetails connectionDetails object
#'
#' @param cdmDatabaseSchema OMOP CDM schema
#' @param cohortDatabaseSchema Cohort schema
#' @param cohortTable Table in cohort database schema with write permissions
#' @param cohortId Unique cohort id
#' @param sql_path Path to the cohort-definition SQL file to render and
#'   execute. Defaults to the package's bundled CreateHFCohort.sql.
#'
#' @export
createCohort <- function(connectionDetails,
                         cdmDatabaseSchema,
                         cohortDatabaseSchema,
                         cohortTable,
                         cohortId,
                         sql_path = system.file("sql/sql_server/CreateHFCohort.sql", package = "HFInYoungWomen")) {
  sql <- readChar(sql_path, file.info(sql_path)$size)
  sql <- SqlRender::render(
    sql = sql,
    vocabulary_database_schema = cdmDatabaseSchema,
    cdm_database_schema = cdmDatabaseSchema,
    target_database_schema = cohortDatabaseSchema,
    target_cohort_table = cohortTable,
    target_cohort_id = cohortId
  )
  sql <- SqlRender::translate(sql, targetDialect = connectionDetails$dbms, tempEmulationSchema = cohortDatabaseSchema)
  connection <- DatabaseConnector::connect(connectionDetails)
  on.exit(DatabaseConnector::disconnect(connection))

  DatabaseConnector::executeSql(connection, sql)
}

