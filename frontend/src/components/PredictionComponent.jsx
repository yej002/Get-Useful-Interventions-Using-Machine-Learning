import React, { useState } from 'react';
import { Chart, ArcElement, Tooltip, Legend } from 'chart.js';
import './PredictionComponent.css';

Chart.register(ArcElement, Tooltip, Legend);

function PredictionComponent() {
  const [trainingFile, setTrainingFile] = useState(null);
  const [predictionFile, setPredictionFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [predictionStatus, setPredictionStatus] = useState('');

  const handleTrainingFileChange = (event) => {
    setTrainingFile(event.target.files[0]);
  };

  const handlePredictionFileChange = (event) => {
    setPredictionFile(event.target.files[0]);
  };

  const handleTrainModel = async () => {
    if (!trainingFile) {
      alert('Please upload a CSV file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', trainingFile);

    try {
      const response = await fetch('http://localhost:8000/train-model/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      alert('Model training initiated.');
      setTrainingStatus('success');
    } catch (error) {
      console.error('Error during file upload for model training:', error);
      alert('Failed to initiate model training.');
      setTrainingStatus('failure');
    }
  };

  const handleMakePrediction = async () => {
    if (!predictionFile) {
      alert('Please upload a CSV file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', predictionFile);

    try {
      const response = await fetch('http://localhost:8000/make-prediction/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPredictionResult(data);
      setPredictionStatus('success');
    } catch (error) {
      console.error('Error during file upload and prediction:', error);
      alert('Failed to make prediction.');
      setPredictionStatus('failure');
    }
  };


  // Helper function to parse and format the JSON result for display
  const formatPredictionResult = (resultData) => {
    try {
      // log out the prediction result in console for validation
      console.log(resultData.result)

      // format the overall probability
      const overallProbabilityDifference = (resultData.overall_probability - resultData.baseline_probability) * 100;
      const formattedOverallProbabilityDifference = `${overallProbabilityDifference >= 0 ? '+' : ''}${overallProbabilityDifference.toFixed(2)}%`;
      const overallTextColor = overallProbabilityDifference > 0 ? 'darkred' : 'black';

      return (
        <div>
          <h2>Probability of Return to Work (Baseline)</h2>
          <pre className="prediction-result">The baseline probability of Return to Work is: {resultData.baseline_probability * 100}%</pre>
          
          <h2>Probability of Return to Work with predicted Interventions</h2>
          <ul>
            {resultData.each_probabilities.map(([category, intervention, probability_return_to_work], index) => {
              // format each of the probability
              const probabilityDifference = (probability_return_to_work - resultData.baseline_probability) * 100;
              const formattedProbabilityDifference = `${probabilityDifference >= 0 ? '+' : ''}${probabilityDifference.toFixed(2)}%`;
              const textColor = probabilityDifference > 0 ? 'darkred' : 'black';

              return (
                <li key={index} className="prediction-result_with_interventions">
                  {category}: Taking <strong>{intervention}</strong>, the probability of Return to Work is: {probability_return_to_work * 100}%. 
                  Changed by <strong><span style={{ color: textColor }}>{formattedProbabilityDifference}</span></strong>
                </li>
              );
            })}            
          </ul>

          <pre className="prediction-result_with_all_interventions">
            The overall probability of Return to Work if taking <strong>ALL</strong> predicted interventions is: {resultData.overall_probability * 100}%.
            Changed by <strong><span style={{ color: overallTextColor }}>{formattedOverallProbabilityDifference}</span></strong>
          </pre>
          

          <h2>Triggered Interventions for Consideration</h2>
          <pre className="triggered_interventions">
            {resultData.triggered_interventions.join('\n')}
          </pre>
        </div>
      );
    } catch (error) {
      console.error('Error formatting prediction result:', error);
      return <p>Error displaying prediction results.</p>;
    }
  };


  return (
    <div className="prediction-component">
      <h1>Get Useful Interventions Using Machine Learning</h1>
      <p>Please select and upload the CSV file containing historical data for model training.{trainingStatus === 'success' ? ' ✅' : trainingStatus === 'failure' ? ' ❌' : ''}</p>
      <div className="file-input-container">
        <input type="file" onChange={handleTrainingFileChange} accept=".csv" />
        <button onClick={handleTrainModel}>Train Model</button>
      </div>

      <p>Please select and upload the CSV file containing target data for prediction.{predictionStatus === 'success' ? ' ✅' : predictionStatus === 'failure' ? ' ❌' : ''}</p>
      <div className="file-input-container">
        <input type="file" onChange={handlePredictionFileChange} accept=".csv" />
        <button onClick={handleMakePrediction}>Make Prediction</button>
      </div>


      {predictionResult && (
        <div>
          {predictionResult && formatPredictionResult(predictionResult)}
        </div>
      )}
    </div>
  );
}

export default PredictionComponent;
