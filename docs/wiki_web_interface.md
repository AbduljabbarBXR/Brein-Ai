# Web Interface Guide

This comprehensive guide covers the Brein AI web interface, its features, customization options, and advanced usage patterns.

## 🌐 Interface Overview

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│                    Brein AI Dashboard                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────┐  ┌─────────────┐ │
│  │ Chat Panel  │  │  Response Area      │  │  Settings   │ │
│  │             │  │                     │  │             │ │
│  │ Input       │  │  Agent Activity     │  │  Theme      │ │
│  │ Field       │  │  Status             │  │  Options    │ │
│  └─────────────┘  └─────────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Memory Stats    │  │ System Health   │  │ Quick       │ │
│  │                 │  │                 │  │ Actions     │ │
│  │ Conversations   │  │ CPU/Memory      │  │             │ │
│  │ Stored          │  │ Usage            │  │ Clear       │ │
│  │ Documents       │  │ Response Time    │  │ Memory      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components
- **Chat Interface**: Main conversation area with message history
- **Agent Status Panel**: Real-time agent activity and coordination
- **System Monitor**: Performance metrics and health indicators
- **Settings Panel**: Customization options and preferences
- **Quick Actions**: Frequently used commands and shortcuts

## 💬 Chat Interface

### Message Types

#### User Messages
```
┌─────────────────────────────────────────────────────────────┐
│ 👤 You                                              2:30 PM │
├─────────────────────────────────────────────────────────────┤
│ Explain how machine learning algorithms work in simple     │
│ terms.                                                     │
└─────────────────────────────────────────────────────────────┘
```

#### AI Responses
```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Brein AI                                        2:30 PM │
├─────────────────────────────────────────────────────────────┤
│ Machine learning algorithms learn patterns from data...    │
│                                                            │
│ 🔍 Agents Used: prefrontal_cortex, hippocampus            │
│ ⏱️  Processing Time: 1.2s                                 │
│ 🎯 Confidence: 92%                                        │
│ 📚 Sources: Internal knowledge                             │
└─────────────────────────────────────────────────────────────┘
```

#### System Messages
```
┌─────────────────────────────────────────────────────────────┐
│ 🔧 System                                         2:31 PM │
├─────────────────────────────────────────────────────────────┤
│ Web content access enabled for this query.                │
│ Retrieved 3 relevant sources from the web.                │
└─────────────────────────────────────────────────────────────┘
```

### Message Features

#### Rich Text Support
- **Markdown Rendering**: Headers, lists, code blocks, links
- **Code Syntax Highlighting**: Support for 100+ programming languages
- **LaTeX Math Rendering**: Mathematical expressions and formulas
- **Table Support**: Data tables and comparison matrices

#### Interactive Elements
- **Copy to Clipboard**: One-click copying of code and responses
- **Regenerate Response**: Request alternative responses
- **Feedback Buttons**: Rate responses (thumbs up/down)
- **Follow-up Suggestions**: AI-generated related questions

## 🎨 Customization Options

### Theme Settings

#### Available Themes
```json
{
  "themes": {
    "light": {
      "background": "#ffffff",
      "surface": "#f8f9fa",
      "text": "#212529",
      "accent": "#007bff"
    },
    "dark": {
      "background": "#1a1a1a",
      "surface": "#2d2d2d",
      "text": "#ffffff",
      "accent": "#4dabf7"
    },
    "auto": {
      "follows_system": true,
      "light_theme": "light",
      "dark_theme": "dark"
    }
  }
}
```

#### Custom CSS
```css
/* Custom theme example */
:root {
  --brein-primary: #6366f1;
  --brein-secondary: #8b5cf6;
  --brein-background: #0f0f23;
  --brein-surface: #1a1a2e;
  --brein-text: #ffffff;
  --brein-border: #16213e;
}

/* Custom chat bubble styling */
.message.user {
  background: linear-gradient(135deg, var(--brein-primary), var(--brein-secondary));
  border-radius: 18px 18px 4px 18px;
}

.message.ai {
  background: var(--brein-surface);
  border: 1px solid var(--brein-border);
  border-radius: 18px 18px 18px 4px;
}
```

### Layout Options

#### Panel Arrangements
- **Classic**: Chat on left, info panels on right
- **Fullscreen**: Chat takes entire screen, panels collapsible
- **Compact**: Minimal interface with essential features only
- **Split**: Chat and response areas side-by-side

#### Responsive Design
- **Desktop**: Full multi-panel layout
- **Tablet**: Collapsible side panels
- **Mobile**: Single-column stacked layout

## ⚙️ Advanced Settings

### Agent Configuration

#### Agent Selection
```javascript
// Manual agent selection
const agentConfig = {
  preferred_agents: ['prefrontal_cortex', 'hippocampus'],
  fallback_agents: ['amygdala'],
  exclude_agents: [], // Agents to never use
  priority_weights: {
    prefrontal_cortex: 0.8,
    hippocampus: 0.6,
    amygdala: 0.4
  }
};
```

#### Custom Agent Prompts
```json
{
  "custom_prompts": {
    "creative_writing": {
      "system_prompt": "You are a creative writing assistant...",
      "temperature": 0.9,
      "max_tokens": 1000
    },
    "technical_analysis": {
      "system_prompt": "You are a technical analyst...",
      "temperature": 0.3,
      "max_tokens": 1500
    }
  }
}
```

### Memory Settings

#### Conversation Management
```javascript
const memorySettings = {
  // Session management
  session_timeout: 3600000, // 1 hour in milliseconds
  max_sessions: 10,
  auto_save: true,

  // Memory limits
  max_conversation_length: 100, // messages per session
  memory_retention_days: 30,
  compression_enabled: true,

  // Search settings
  search_similarity_threshold: 0.7,
  max_search_results: 20,
  include_metadata: true
};
```

#### Data Export/Import
```javascript
// Export conversation data
async function exportConversation(sessionId) {
  const response = await fetch(`/api/conversations/${sessionId}/export`);
  const data = await response.json();
  return data;
}

// Import conversation data
async function importConversation(data) {
  const response = await fetch('/api/conversations/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return response.json();
}
```

## 📊 Real-Time Monitoring

### System Health Dashboard

#### Performance Metrics
```
┌─────────────────────────────────────────────────────────────┐
│                    System Performance                       │
├─────────────────────────────────────────────────────────────┤
│  Response Time: 1.2s (avg)  •  CPU: 45%  •  Memory: 2.1GB  │
│  Active Sessions: 12  •  Queue Length: 3  •  Uptime: 99.9% │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Agent Status    │  │ Memory Usage     │  │ Network     │ │
│  │ ✅ PFC Active   │  │ 1.2GB / 4GB     │  │ 15ms latency │ │
│  │ ✅ HIP Active   │  │ 234 docs stored  │  │ 98% success │ │
│  │ ✅ AMY Active   │  │ 45MB cache       │  │ rate         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### Agent Activity Monitor
```javascript
class AgentMonitor {
  constructor() {
    this.agents = ['prefrontal_cortex', 'hippocampus', 'amygdala'];
    this.status = {};
    this.updateInterval = 5000; // 5 seconds
  }

  async updateStatus() {
    for (const agent of this.agents) {
      try {
        const response = await fetch(`/api/agents/${agent}/status`);
        const status = await response.json();
        this.status[agent] = status;
        this.updateUI(agent, status);
      } catch (error) {
        this.status[agent] = { status: 'error', error: error.message };
      }
    }
  }

  updateUI(agent, status) {
    const indicator = document.getElementById(`${agent}-status`);
    indicator.className = `status-indicator ${status.status}`;

    if (status.status === 'active') {
      indicator.innerHTML = '🟢 Active';
    } else if (status.status === 'busy') {
      indicator.innerHTML = '🟡 Busy';
    } else {
      indicator.innerHTML = '🔴 Error';
    }
  }

  startMonitoring() {
    this.updateStatus();
    setInterval(() => this.updateStatus(), this.updateInterval);
  }
}
```

## 🔧 Developer Features

### API Integration

#### REST API Client
```javascript
class BreinAPIClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.sessionId = null;
  }

  async query(text, options = {}) {
    const response = await fetch(`${this.baseURL}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        query: text,
        session_id: this.sessionId,
        ...options
      })
    });

    const result = await response.json();
    this.sessionId = result.session_id;
    return result;
  }

  async getMemoryStats() {
    const response = await fetch(`${this.baseURL}/api/memory/stats`);
    return response.json();
  }

  async exportData() {
    const response = await fetch(`${this.baseURL}/api/user/export-data`);
    return response.json();
  }
}

// Usage
const client = new BreinAPIClient();
const result = await client.query("Hello, how are you?");
console.log(result.response);
```

#### WebSocket Integration
```javascript
class BreinWebSocketClient {
  constructor(url = 'ws://localhost:8000/ws') {
    this.url = url;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = (event) => {
        console.log('Connected to Brein AI');
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };

      this.ws.onclose = (event) => {
        console.log('Disconnected from Brein AI');
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to connect:', error);
    }
  }

  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  handleMessage(data) {
    switch (data.type) {
      case 'response':
        this.displayResponse(data);
        break;
      case 'agent_status':
        this.updateAgentStatus(data);
        break;
      case 'system_health':
        this.updateSystemHealth(data);
        break;
    }
  }

  handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, delay);
    }
  }
}
```

### Custom Plugins

#### Plugin Architecture
```javascript
// Plugin interface
class BreinPlugin {
  constructor() {
    this.name = 'Custom Plugin';
    this.version = '1.0.0';
    this.description = 'Custom functionality for Brein AI';
  }

  // Lifecycle methods
  async initialize() {
    // Plugin initialization
  }

  async destroy() {
    // Cleanup resources
  }

  // Hook methods
  onMessageSend(message) {
    // Modify outgoing messages
    return message;
  }

  onMessageReceive(message) {
    // Modify incoming messages
    return message;
  }

  onUIUpdate(element) {
    // Modify UI elements
    return element;
  }
}

// Custom theme plugin example
class DarkThemePlugin extends BreinPlugin {
  constructor() {
    super();
    this.name = 'Dark Theme Plugin';
  }

  onUIUpdate(element) {
    if (element.classList.contains('message')) {
      element.style.backgroundColor = '#2d2d2d';
      element.style.color = '#ffffff';
    }
    return element;
  }
}

// Register plugin
window.breinPlugins = window.breinPlugins || [];
window.breinPlugins.push(new DarkThemePlugin());
```

## 📱 Mobile Responsiveness

### Responsive Breakpoints
```css
/* Mobile (up to 768px) */
@media (max-width: 768px) {
  .chat-container {
    flex-direction: column;
  }

  .side-panels {
    display: none;
  }

  .mobile-menu-toggle {
    display: block;
  }

  .message {
    max-width: 100%;
    margin: 8px;
  }
}

/* Tablet (769px to 1024px) */
@media (min-width: 769px) and (max-width: 1024px) {
  .chat-container {
    flex-direction: row;
  }

  .side-panels {
    width: 300px;
  }

  .main-chat {
    flex: 1;
  }
}

/* Desktop (1025px+) */
@media (min-width: 1025px) {
  .chat-container {
    max-width: 1400px;
    margin: 0 auto;
  }

  .side-panels {
    width: 350px;
  }
}
```

### Touch Interactions
```javascript
class TouchHandler {
  constructor(element) {
    this.element = element;
    this.touchStartX = 0;
    this.touchStartY = 0;
    this.bindEvents();
  }

  bindEvents() {
    this.element.addEventListener('touchstart', this.handleTouchStart.bind(this));
    this.element.addEventListener('touchmove', this.handleTouchMove.bind(this));
    this.element.addEventListener('touchend', this.handleTouchEnd.bind(this));
  }

  handleTouchStart(event) {
    this.touchStartX = event.touches[0].clientX;
    this.touchStartY = event.touches[0].clientY;
  }

  handleTouchMove(event) {
    if (!this.touchStartX || !this.touchStartY) return;

    const touchEndX = event.touches[0].clientX;
    const touchEndY = event.touches[0].clientY;

    const deltaX = touchEndX - this.touchStartX;
    const deltaY = touchEndY - this.touchStartY;

    // Handle swipe gestures
    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      if (deltaX > 50) {
        this.handleSwipeRight();
      } else if (deltaX < -50) {
        this.handleSwipeLeft();
      }
    }
  }

  handleTouchEnd() {
    this.touchStartX = 0;
    this.touchStartY = 0;
  }

  handleSwipeRight() {
    // Show side panel
    document.querySelector('.side-panels').classList.add('visible');
  }

  handleSwipeLeft() {
    // Hide side panel
    document.querySelector('.side-panels').classList.remove('visible');
  }
}

// Initialize touch handling
new TouchHandler(document.querySelector('.chat-container'));
```

## 🔒 Security Features

### Authentication Integration
```javascript
class AuthManager {
  constructor() {
    this.isAuthenticated = false;
    this.user = null;
    this.token = localStorage.getItem('brein_token');
  }

  async login(username, password) {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();

      if (response.ok) {
        this.token = data.token;
        this.user = data.user;
        this.isAuthenticated = true;

        localStorage.setItem('brein_token', this.token);
        this.updateUI();
      } else {
        throw new Error(data.message);
      }
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  }

  async logout() {
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${this.token}` }
      });
    } finally {
      this.token = null;
      this.user = null;
      this.isAuthenticated = false;
      localStorage.removeItem('brein_token');
      this.updateUI();
    }
  }

  updateUI() {
    const authElements = document.querySelectorAll('.auth-required');
    authElements.forEach(element => {
      element.style.display = this.isAuthenticated ? 'block' : 'none';
    });
  }
}
```

### Privacy Controls
```javascript
class PrivacyManager {
  constructor() {
    this.settings = {
      storeMessages: true,
      allowWebAccess: false,
      shareAnalytics: false,
      dataRetentionDays: 30
    };
    this.loadSettings();
  }

  loadSettings() {
    const saved = localStorage.getItem('brein_privacy');
    if (saved) {
      this.settings = { ...this.settings, ...JSON.parse(saved) };
    }
  }

  saveSettings() {
    localStorage.setItem('brein_privacy', JSON.stringify(this.settings));
  }

  updateSetting(key, value) {
    this.settings[key] = value;
    this.saveSettings();
    this.applySettings();
  }

  applySettings() {
    // Apply privacy settings to the interface
    if (!this.settings.storeMessages) {
      // Disable message storage
      document.querySelectorAll('.message-store-toggle').forEach(el => {
        el.checked = false;
      });
    }

    if (!this.settings.allowWebAccess) {
      // Hide web access options
      document.querySelectorAll('.web-access-option').forEach(el => {
        el.style.display = 'none';
      });
    }
  }

  exportData() {
    // Export user's data
    return {
      messages: this.getStoredMessages(),
      preferences: this.settings,
      exportDate: new Date().toISOString()
    };
  }

  deleteData() {
    // Delete all user data
    localStorage.removeItem('brein_privacy');
    localStorage.removeItem('brein_token');
    // Additional cleanup...
  }
}
```

## 🎨 Accessibility Features

### Keyboard Navigation
```javascript
class KeyboardNavigation {
  constructor() {
    this.focusableElements = [];
    this.currentFocusIndex = 0;
    this.bindEvents();
  }

  bindEvents() {
    document.addEventListener('keydown', this.handleKeydown.bind(this));
    this.updateFocusableElements();
  }

  updateFocusableElements() {
    this.focusableElements = Array.from(
      document.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      )
    ).filter(el => !el.disabled && el.offsetParent !== null);
  }

  handleKeydown(event) {
    switch (event.key) {
      case 'Tab':
        event.preventDefault();
        if (event.shiftKey) {
          this.focusPrevious();
        } else {
          this.focusNext();
        }
        break;

      case 'Enter':
        if (document.activeElement === document.querySelector('#message-input')) {
          this.sendMessage();
        }
        break;

      case 'Escape':
        this.closeModals();
        break;
    }
  }

  focusNext() {
    this.currentFocusIndex = (this.currentFocusIndex + 1) % this.focusableElements.length;
    this.focusableElements[this.currentFocusIndex].focus();
  }

  focusPrevious() {
    this.currentFocusIndex = (this.currentFocusIndex - 1 + this.focusableElements.length) % this.focusableElements.length;
    this.focusableElements[this.currentFocusIndex].focus();
  }

  sendMessage() {
    const input = document.querySelector('#message-input');
    const message = input.value.trim();
    if (message) {
      // Send message logic
      input.value = '';
    }
  }

  closeModals() {
    document.querySelectorAll('.modal.open').forEach(modal => {
      modal.classList.remove('open');
    });
  }
}

// Initialize keyboard navigation
new KeyboardNavigation();
```

### Screen Reader Support
```html
<!-- Accessible chat interface -->
<div role="main" aria-label="Brein AI Chat Interface">
  <div role="log" aria-label="Chat History" aria-live="polite" aria-atomic="false">
    <div role="article" aria-label="User message" class="message user">
      <span class="sender" aria-hidden="true">👤 You</span>
      <div class="content">Hello, how are you?</div>
      <time datetime="2025-11-12T14:30:00" aria-label="Sent at 2:30 PM">2:30 PM</time>
    </div>

    <div role="article" aria-label="AI response" class="message ai">
      <span class="sender" aria-hidden="true">🤖 Brein AI</span>
      <div class="content">I'm doing well, thank you for asking!</div>
      <div class="metadata" aria-label="Response metadata">
        <span aria-label="Processing time: 1.2 seconds">⏱️ 1.2s</span>
        <span aria-label="Confidence: 95%">🎯 95%</span>
      </div>
      <time datetime="2025-11-12T14:30:02" aria-label="Responded at 2:30 PM">2:30 PM</time>
    </div>
  </div>

  <form role="form" aria-label="Send Message">
    <label for="message-input" class="sr-only">Type your message</label>
    <textarea
      id="message-input"
      aria-describedby="input-help"
      aria-invalid="false"
      placeholder="Type your message..."
    ></textarea>
    <div id="input-help" class="sr-only">
      Press Enter to send your message, or Shift+Enter for a new line
    </div>
    <button type="submit" aria-label="Send message">
      <span aria-hidden="true">📤</span>
      Send
    </button>
  </form>
</div>
```

## 📚 Related Documentation

- [[Quick Start|Quick-Start]] - Get started with the web interface
- [[User Manual|User-Manual]] - Complete user guide
- [[API Reference|API-Reference]] - Technical API details
- [[Configuration|Configuration]] - Interface customization options

---

*Web Interface Guide - Last updated: November 2025*
