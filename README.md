# ClotSense - A Bloot Clot Origin identification system for Ischemic Stroke

## **Overview**

Classification of stroke blood clot origin is a crucial step in the diagnosis and treatment of ischemic stroke. This project is aimed at developing an automated system that can accurately identify the origin of blood clots in patients with ischemic stroke. The system uses state-of-the-art deep learning algorithms to analyze whole slide digital pathology image to classify the type of blood clot that caused the stroke. This project can classify between **Cardioembolic** and **Large Artery Atherosclerosis** blood clots with an **f1-score** of **96.3**.

## **How it works?**

 <p align="center"> <img src="https://github.com/Koushik0901/Stroke-Blood-Clot-Origin-Prediction/blob/main/app/static/img/how_it_works.gif" width="700" height="400"  />
</p>

## **Running on native machine**
* clone the repository using `git clone https://github.com/Koushik0901/Stroke-Blood-Clot-Origin-Prediction.git`
* install [docker](https://www.docker.com/)
* execute `docker compose up` on your terminal
* open `localhost:5000` on your browser to view the running application
