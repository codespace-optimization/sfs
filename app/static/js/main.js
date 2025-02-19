document.getElementById('promptForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const model = document.getElementById('model').value;
    const apiKey = document.getElementById('apiKey').value;
    const prompt = document.getElementById('prompt').value;
    const numTries = parseInt(document.getElementById('numTries').value, 10);

    // Show loading screen with dynamic text
    const loadingText = document.getElementById('loadingText');
    const loadingContainer = document.getElementById('loading');
    loadingContainer.style.display = 'block';

    const loadingMessages = [
        "Thinking",
        "Generating unit tests",
        "Generating new seed solution",
        "Testing out solution on unit tests",
        "Planning out next steps",
        "Revising solution",
        "Summarizing insights"
    ];

    let loadingIndex = 0;
    let loadingInterval = setInterval(() => {
        loadingText.innerText = loadingMessages[loadingIndex];
        loadingIndex = (loadingIndex + 1) % loadingMessages.length;
    }, 1000); // Change every 1 second

    document.getElementById('output').innerHTML = ''; // Clear previous output

    try {
        const response = await fetch('/process_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, apiKey, prompt, numTries })
        });

        const result = await response.json();
        
        clearInterval(loadingInterval); // Stop loading animation
        loadingContainer.style.display = 'none'; // Hide loading container

        if (result.error) {
            document.getElementById('output').innerText = result.error;
        } else {
            // Extract output values
            const outputText = result.output || "No output generated.";
            const numTestsPassed = result.num_tests_passed || 0;
            const numTestsGenerated = result.num_tests_generated || 1; // Prevent division by zero
            const numRevisions = result.num_revisions || 0;

            // Construct output string
            const outputHTML = `
                <p><strong>Output:</strong> ${outputText}</p>
                <p><strong>Tests Passed:</strong> ${numTestsPassed} / ${numTestsGenerated}</p>
                <p><strong>Number of Revisions:</strong> ${numRevisions}</p>
            `;

            document.getElementById('output').innerHTML = outputHTML;
        }
    } catch (error) {
        clearInterval(loadingInterval);
        document.getElementById('output').innerText = 'An error occurred. Please try again.';
    } finally {
        loadingContainer.style.display = 'none'; // Hide loading container
    }
});
