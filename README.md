I built an age classification model using PyTorch. For the architecture, I used a ResNet50 backbone because it provides a good balance between depth and performance. The dataset was organized in folders named after assessment_excercise_test (e.g., "20", "21", etc.), so it was easy to extract age labels.
All input images were resized to 224×224 and normalized using standard ImageNet mean and standard deviation. I split the data into 80% training and 20% validation. The model was trained for 25 epochs, but training took a lot of time due to limited hardware.


![Screenshot 2025-06-21 082456](https://github.com/user-attachments/assets/1568eb30-11fa-4202-b587-290c88c49b20)
![Screenshot 2025-06-21 082511](https://github.com/user-attachments/assets/50ea23cb-8b12-4981-ab60-4595ad94d480)
![Screenshot 2025-06-21 082623](https://github.com/user-attachments/assets/c4c2b2c6-d603-41ec-84ed-ad0a44ddad56)
![Screenshot 2025-06-21 082859](https://github.com/user-attachments/assets/91c02c1e-1535-4216-a7b6-6a2f257d30f1)
![Screenshot 2025-06-21 082748](https://github.com/user-attachments/assets/ade20b88-97ae-45af-8e58-aa739ade49f9)
