// ===================================
// STORAGE MANAGER - LocalStorage Wrapper
// ===================================

class StorageManager {
    constructor() {
        this.STORAGE_KEYS = {
            THEME: 'miniTherapist_theme',
            CONVERSATIONS: 'miniTherapist_conversations',
            SESSION_ID: 'miniTherapist_sessionId',
            MOOD_HISTORY: 'miniTherapist_moodHistory'
        };
    }

    // Theme Management
    getTheme() {
        return localStorage.getItem(this.STORAGE_KEYS.THEME) || 'light';
    }

    setTheme(theme) {
        localStorage.setItem(this.STORAGE_KEYS.THEME, theme);
        document.documentElement.setAttribute('data-theme', theme);
    }

    // Session Management
    getSessionId() {
        let sessionId = localStorage.getItem(this.STORAGE_KEYS.SESSION_ID);
        if (!sessionId) {
            sessionId = this._generateSessionId();
            localStorage.setItem(this.STORAGE_KEYS.SESSION_ID, sessionId);
        }
        return sessionId;
    }

    _generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Conversation Management
    getConversations() {
        const data = localStorage.getItem(this.STORAGE_KEYS.CONVERSATIONS);
        return data ? JSON.parse(data) : [];
    }

    saveConversation(message) {
        const conversations = this.getConversations();
        conversations.push({
            ...message,
            timestamp: new Date().toISOString()
        });

        // Keep only last 50 messages
        if (conversations.length > 50) {
            conversations.shift();
        }

        localStorage.setItem(this.STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
    }

    clearConversations() {
        localStorage.removeItem(this.STORAGE_KEYS.CONVERSATIONS);
    }

    // Mood History Management
    getMoodHistory() {
        const data = localStorage.getItem(this.STORAGE_KEYS.MOOD_HISTORY);
        return data ? JSON.parse(data) : [];
    }

    saveMood(emotion, confidence) {
        const moodHistory = this.getMoodHistory();
        moodHistory.push({
            emotion,
            confidence,
            timestamp: new Date().toISOString()
        });

        // Keep only last 30 moods
        if (moodHistory.length > 30) {
            moodHistory.shift();
        }

        localStorage.setItem(this.STORAGE_KEYS.MOOD_HISTORY, JSON.stringify(moodHistory));
    }

    clearMoodHistory() {
        localStorage.removeItem(this.STORAGE_KEYS.MOOD_HISTORY);
    }

    // Clear all data
    clearAll() {
        Object.values(this.STORAGE_KEYS).forEach(key => {
            if (key !== this.STORAGE_KEYS.THEME) {
                localStorage.removeItem(key);
            }
        });
    }
}

// Export for use in other scripts
const storage = new StorageManager();
