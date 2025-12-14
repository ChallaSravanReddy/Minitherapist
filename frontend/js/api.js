// ===================================
// API CLIENT - Backend Communication
// ===================================

class APIClient {
    constructor(baseURL = 'http://localhost:5000') {
        this.baseURL = baseURL;
        this.sessionId = storage.getSessionId();
    }

    async _fetch(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    // Send chat message
    async sendMessage(message) {
        return await this._fetch('/api/chat', {
            method: 'POST',
            body: JSON.stringify({
                message,
                session_id: this.sessionId
            })
        });
    }

    // Get daily affirmation
    async getAffirmation() {
        return await this._fetch('/api/affirmation');
    }

    // Get spiritual quote
    async getQuote() {
        return await this._fetch('/api/quote');
    }

    // Get mood history
    async getMoodHistory() {
        return await this._fetch('/api/mood-history', {
            method: 'POST',
            body: JSON.stringify({
                session_id: this.sessionId
            })
        });
    }

    // Health check
    async healthCheck() {
        return await this._fetch('/api/health');
    }
}

// Export for use in other scripts
const api = new APIClient();
