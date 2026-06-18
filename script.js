document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    
    // UI States
    const initialState = document.getElementById('initial-state');
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    const successState = document.getElementById('success-state');
    
    // Elements to update
    const resultBadge = document.getElementById('result-badge');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const probValue = document.getElementById('prob-value');
    const probTextVal = document.getElementById('prob-text-val');
    const gaugePath = document.getElementById('gauge-path');
    const errorMessage = document.getElementById('error-message');
    const historyBody = document.getElementById('history-body');
    
    // History session storage array
    let sessionHistory = [];
    
    function updateHistoryTable() {
        if (sessionHistory.length === 0) {
            historyBody.innerHTML = '<tr class="empty-row"><td colspan="5">No predictions made yet in this session.</td></tr>';
            return;
        }
        
        let html = '';
        sessionHistory.forEach(item => {
            const badgeClass = item.probability >= 50 ? 'hist-green' : 'hist-red';
            html += `
                <tr>
                    <td>${item.age}</td>
                    <td>€${parseFloat(item.balance).toLocaleString()}</td>
                    <td style="text-transform: capitalize;">${item.job}</td>
                    <td>${item.probability}%</td>
                    <td><span class="hist-badge ${badgeClass}">${item.result}</span></td>
                </tr>
            `;
        });
        historyBody.innerHTML = html;
    }
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide all states, show loading
        initialState.classList.add('hidden');
        errorState.classList.add('hidden');
        successState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        // Disable button
        const submitBtn = document.getElementById('predict-btn');
        submitBtn.disabled = true;
        
        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            loadingState.classList.add('hidden');
            
            if (response.ok && result.success) {
                // Update UI with result
                successState.classList.remove('hidden');
                
                resultText.textContent = result.result;
                probValue.textContent = `${result.probability}%`;
                probTextVal.textContent = `${result.probability}%`;
                
                // Styling based on prediction
                resultBadge.className = 'result-badge'; // reset
                gaugePath.className = 'circle'; // reset
                probValue.className = 'percentage'; // reset
                
                if (result.probability >= 50) {
                    resultBadge.classList.add('badge-success');
                    resultIcon.className = 'fa-solid fa-check';
                    gaugePath.classList.add('gauge-success');
                    probValue.classList.add('text-success');
                } else {
                    resultBadge.classList.add('badge-danger');
                    resultIcon.className = 'fa-solid fa-xmark';
                    gaugePath.classList.add('gauge-danger');
                    probValue.classList.add('text-danger');
                }
                
                // Animate circular gauge SVG
                setTimeout(() => {
                    gaugePath.setAttribute('stroke-dasharray', `${result.probability}, 100`);
                }, 100);
                
                // Update history
                sessionHistory.unshift({
                    age: data.age,
                    balance: data.balance,
                    job: data.job,
                    probability: result.probability,
                    result: result.result
                });
                
                if (sessionHistory.length > 5) {
                    sessionHistory.pop();
                }
                
                updateHistoryTable();
                
            } else {
                // Show error state from server
                errorState.classList.remove('hidden');
                errorMessage.textContent = result.error || 'Prediction failed.';
            }
            
        } catch (error) {
            console.error('Error:', error);
            loadingState.classList.add('hidden');
            errorState.classList.remove('hidden');
            errorMessage.textContent = 'A network error occurred. Ensure the server is running.';
        } finally {
            submitBtn.disabled = false;
        }
    });
});
