CREATE KEYSPACE IF NOT EXISTS space 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE IF NOT EXISTS space.data (
    "Year" INT,
    "Title" TEXT,
    "PublicationName" TEXT,
    "CitedByCount" INT,
    "AffiliationCountry" TEXT,
    "AuthorKeywords" TEXT,
    PRIMARY KEY ("Year", "Title")
);