CREATE DATABASE clotDB;
USE clotDB;
CREATE TABLE `users` (
  `email` varchar(256) NOT NULL,
  `password` varchar(256) NOT NULL,
  PRIMARY KEY (`email`)
);
LOCK TABLES `users` WRITE;
INSERT INTO `users`
VALUES ('k@gmail.com', 'pbkdf2:sha256:260000$nRZddkrgdmjfCKjQ$507d15f4f51c202cc4a006127d4355ec7abef63e57fadcc9358f15289202810e'),
  ('koushik.nov01@gmail.com', 'pbkdf2:sha256:260000$MwR0XcNAOMCvhMnO$dd9c5e1571ce7877237e8e6ec16dd198b224b4274385f633ba66ad3b06258a58'),
  ('mahilogesh57@gmail.com', 'pbkdf2:sha256:260000$VU5jGxQlhSELyVDm$2ea674373cb7fa803df421265eaf1decbc9be9742fe4458deb5d1d0a0dcee516');
UNLOCK TABLES;