1.store all file in nlp_pipeline folder

2.come to my room and collect mymodel file.

3. create a models folder inside of predict label folder.

4.in the models folder store my_model folder , label_encoder.pkl file.

5.then in the vs code terminal nlp_pipeline/-:

5.1.python -m venv myenv
   
5.2.myenv\scripts\activate
   
5.3.pip install -r requirements.txt 
   
5.4.in one terminal run 

5.4.1.cd predict_labels

5.4.2.uvicorn backend:app --reload --port 8000
       
5.5.in another terminal run 

5.5.1.cd predict_labels

5.5.2.streamlit run frontend.py
       
5.6.in another terminal run 

5.6.1.cd sentiment_analysis

5.6.2.uvicorn backend:app --reload --port 8500

5.7.in another terminal run 

5.7.1.cd sentiment_analysis

5.7.2.streamlit run frontend.py

6.then go to

6.1.sentiment analysis localhost frontend website it is your working file.

for api_key- contact to Mahim.....
      
