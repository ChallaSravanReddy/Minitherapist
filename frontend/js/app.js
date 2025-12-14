// ===================================
// MAIN APPLICATION LOGIC
// ===================================

class MiniTherapistApp {
    constructor() {
        this.initializeElements();
        this.initializeTheme();
        this.attachEventListeners();
        this.loadConversationHistory();
        this.checkBackendHealth();
    }

    initializeElements() {
        // Chat elements
        this.messagesContainer = document.getElementById('messages');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.loadingIndicator = document.getElementById('loadingIndicator');

        // Header buttons
        this.themeToggle = document.getElementById('themeToggle');
        this.affirmationBtn = document.getElementById('affirmationBtn');
        this.quoteBtn = document.getElementById('quoteBtn');
        this.moodBtn = document.getElementById('moodBtn');

        // Modals
        this.affirmationModal = document.getElementById('affirmationModal');
        this.quoteModal = document.getElementById('quoteModal');
        this.moodModal = document.getElementById('moodModal');

        // Modal content
        this.affirmationText = document.getElementById('affirmationText');
        this.quoteText = document.getElementById('quoteText');
        this.quoteAuthor = document.getElementById('quoteAuthor');
        this.quoteTradition = document.getElementById('quoteTradition');
    }

    initializeTheme() {
        const theme = storage.getTheme();
        storage.setTheme(theme);
        this.updateThemeIcon(theme);
    }

    updateThemeIcon(theme) {
        const icon = this.themeToggle.querySelector('.theme-icon');
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    attachEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = this.userInput.scrollHeight + 'px';
        });

        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // Modals
        this.affirmationBtn.addEventListener('click', () => this.showAffirmation());
        this.quoteBtn.addEventListener('click', () => this.showQuote());
        this.moodBtn.addEventListener('click', () => this.showMoodTracker());

        // Close modals
        document.getElementById('closeAffirmationModal').addEventListener('click', () => {
            this.affirmationModal.classList.remove('active');
        });
        document.getElementById('closeQuoteModal').addEventListener('click', () => {
            this.quoteModal.classList.remove('active');
        });
        document.getElementById('closeMoodModal').addEventListener('click', () => {
            this.moodModal.classList.remove('active');
        });

        // Refresh buttons
        document.getElementById('refreshAffirmation').addEventListener('click', () => this.showAffirmation());
        document.getElementById('refreshQuote').addEventListener('click', () => this.showQuote());

        // Close modals on background click
        [this.affirmationModal, this.quoteModal, this.moodModal].forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.classList.remove('active');
                }
            });
        });
    }

    toggleTheme() {
        const currentTheme = storage.getTheme();
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        storage.setTheme(newTheme);
        this.updateThemeIcon(newTheme);
    }

    async sendMessage() {
        const message = this.userInput.value.trim();

        if (!message) return;

        // Disable input
        this.userInput.disabled = true;
        this.sendBtn.disabled = true;

        // Add user message to UI
        this.addMessage(message, 'user');

        // Clear input
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        // Show loading
        this.showLoading();

        try {
            // Send to backend
            const response = await api.sendMessage(message);

            // Hide loading
            this.hideLoading();

            // Add bot response
            this.addMessage(response.response, 'bot', {
                emotion: response.emotion,
                confidence: response.confidence,
                isCrisis: response.is_crisis
            });

            // Save to storage
            storage.saveConversation({
                user: message,
                bot: response.response,
                emotion: response.emotion
            });

            // Save mood if not crisis
            if (!response.is_crisis) {
                storage.saveMood(response.emotion, response.confidence);
            }

        } catch (error) {
            this.hideLoading();
            this.addMessage(
                "I'm having trouble connecting right now. Please make sure the backend server is running. You can start it with: python backend/app.py",
                'bot',
                { emotion: 'error' }
            );
            console.error('Error sending message:', error);
        }

        // Re-enable input
        this.userInput.disabled = false;
        this.sendBtn.disabled = false;
        this.userInput.focus();
    }

    addMessage(text, sender, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? '👤' : '🧠';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;

        // Add emotion badge for bot messages
        if (sender === 'bot' && metadata.emotion && metadata.emotion !== 'error') {
            const badge = document.createElement('div');
            badge.className = 'emotion-badge';
            badge.textContent = metadata.isCrisis ? '🆘 Crisis Support' : `💭 ${metadata.emotion}`;
            bubble.appendChild(badge);
        }

        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.appendChild(avatar);
        const contentDiv = document.createElement('div');
        contentDiv.appendChild(bubble);
        contentDiv.appendChild(time);
        messageDiv.appendChild(contentDiv);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showLoading() {
        this.loadingIndicator.classList.add('active');
        this.messagesContainer.appendChild(this.loadingIndicator);
        this.scrollToBottom();
    }

    hideLoading() {
        this.loadingIndicator.classList.remove('active');
    }

    scrollToBottom() {
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    loadConversationHistory() {
        const conversations = storage.getConversations();

        // Load last 10 conversations
        const recent = conversations.slice(-10);
        recent.forEach(conv => {
            this.addMessage(conv.user, 'user');
            this.addMessage(conv.bot, 'bot', { emotion: conv.emotion });
        });
    }

    async showAffirmation() {
        this.affirmationModal.classList.add('active');
        this.affirmationText.textContent = 'Loading...';

        try {
            const response = await api.getAffirmation();
            this.affirmationText.textContent = response.affirmation;
        } catch (error) {
            this.affirmationText.textContent = 'Unable to load affirmation. Please try again.';
            console.error('Error loading affirmation:', error);
        }
    }

    async showQuote() {
        this.quoteModal.classList.add('active');
        this.quoteText.textContent = 'Loading...';
        this.quoteAuthor.textContent = '';
        this.quoteTradition.textContent = '';

        try {
            const response = await api.getQuote();
            this.quoteText.textContent = `"${response.quote}"`;
            this.quoteAuthor.textContent = `— ${response.author}`;
            this.quoteTradition.textContent = response.tradition;
        } catch (error) {
            this.quoteText.textContent = 'Unable to load quote. Please try again.';
            console.error('Error loading quote:', error);
        }
    }

    showMoodTracker() {
        this.moodModal.classList.add('active');
        this.renderMoodStats();
    }

    renderMoodStats() {
        const moodHistory = storage.getMoodHistory();
        const statsContainer = document.getElementById('moodStats');

        if (moodHistory.length === 0) {
            statsContainer.innerHTML = '<p style="text-align: center; color: var(--text-muted);">No mood data yet. Start chatting to track your emotions!</p>';
            return;
        }

        // Count emotions
        const emotionCounts = {};
        moodHistory.forEach(mood => {
            emotionCounts[mood.emotion] = (emotionCounts[mood.emotion] || 0) + 1;
        });

        // Get most common emotion
        const mostCommon = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0];

        // Render stats
        statsContainer.innerHTML = `
            <div class="mood-stat">
                <div class="mood-stat-value">${moodHistory.length}</div>
                <div class="mood-stat-label">Total Conversations</div>
            </div>
            <div class="mood-stat">
                <div class="mood-stat-value">${mostCommon[0]}</div>
                <div class="mood-stat-label">Most Common Emotion</div>
            </div>
            <div class="mood-stat">
                <div class="mood-stat-value">${Object.keys(emotionCounts).length}</div>
                <div class="mood-stat-label">Unique Emotions</div>
            </div>
        `;

        // Simple emotion list
        const emotionList = document.createElement('div');
        emotionList.style.marginTop = 'var(--spacing-lg)';
        emotionList.innerHTML = '<h3 style="margin-bottom: var(--spacing-md);">Emotion Breakdown:</h3>';

        Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]).forEach(([emotion, count]) => {
            const percentage = ((count / moodHistory.length) * 100).toFixed(1);
            emotionList.innerHTML += `
                <div style="display: flex; justify-content: space-between; padding: var(--spacing-sm); background: var(--bg-glass); border-radius: var(--radius-sm); margin-bottom: var(--spacing-xs);">
                    <span style="text-transform: capitalize;">${emotion}</span>
                    <span style="font-weight: 600;">${count} (${percentage}%)</span>
                </div>
            `;
        });

        statsContainer.appendChild(emotionList);
    }

    async checkBackendHealth() {
        try {
            const health = await api.healthCheck();
            console.log('✓ Backend connected:', health);
        } catch (error) {
            console.warn('⚠ Backend not available. Please start the server with: python backend/app.py');
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new MiniTherapistApp();
    console.log('🧠 Mini Therapist initialized');
});
