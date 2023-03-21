CREATE DATABASE clotDB;
USE clotDB;
CREATE TABLE `users` (
  `email` varchar(150) NOT NULL,
  `password` varchar(50) NOT NULL,
  PRIMARY KEY (`email`)
);
LOCK TABLES `users` WRITE;
INSERT INTO `users`
VALUES ('k@gmail.com', 'k'),
  ('koushik.nov01@gmail.com', 'koushik'),
  ('mahilogesh57@gmail.com', 'logesh');
UNLOCK TABLES;