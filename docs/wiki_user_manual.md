# User Manual

This comprehensive user manual covers everything you need to know to effectively use Brein AI for your daily tasks, research, and creative projects.

## 👋 Introduction

### What is Brein AI?
Brein AI is a revolutionary memory-first artificial intelligence system that combines:

- **Persistent Memory**: Remembers conversations and learned information across sessions
- **Multi-Agent Reasoning**: Four specialized AI agents work together for comprehensive responses
- **Web Integration**: Access current information when needed
- **Safety-First Design**: Built-in content filtering and human oversight

### Key Features for Users
- **Conversational Memory**: Remembers your preferences, facts, and conversation history
- **Multi-Modal Responses**: Combines reasoning, emotion, and factual knowledge
- **Web-Aware**: Can fetch current information with your permission
- **Privacy-Focused**: You control what gets stored and accessed

## 💬 Getting Started with Conversations

### Your First Conversation
1. Open Brein AI in your browser: `http://localhost:8000`
2. Type your message in the chat box
3. Press Enter or click Send
4. Brein AI will respond with context-aware answers

### Understanding Response Metadata
Each response includes helpful information:
- **Processing Time**: How long it took to generate the response
- **Agents Used**: Which AI agents contributed to the answer
- **Confidence Score**: How confident the system is in its response
- **Sources**: Web sources used (if web access was enabled)

## 🧠 Working with Memory

### Teaching Brein AI
```bash
# Teach specific facts
"I work as a software developer at TechCorp"
"My favorite coffee is Ethiopian single-origin"
"I prefer Python over JavaScript for backend development"
```

### Memory Recall
```bash
# Ask about remembered information
"What do I do for work?"
"What's my favorite type of coffee?"
"What programming languages do I prefer?"
```

### Memory Management
```bash
# View memory statistics
curl http://localhost:8000/api/memory/stats

# Search your memory
curl "http://localhost:8000/api/memory/search?q=programming"
```

## 🔍 Advanced Query Techniques

### Asking Good Questions

#### Be Specific
```
❌ "Tell me about AI"
✅ "Explain how neural networks work in machine learning"
```

#### Provide Context
```
❌ "What's the best approach?"
✅ "I'm building a web app with React and Node.js. What's the best approach for user authentication?"
```

#### Ask for Reasoning
```
❌ "Should I use MongoDB?"
✅ "I'm building an e-commerce site. Should I use MongoDB or PostgreSQL for the product catalog? Explain the trade-offs."
```

### Using Web Access
```bash
# Enable web access for current information
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in quantum computing?",
    "enable_web_access": true
  }'
```

**Note**: Web access requires explicit permission and goes through safety filters.

## 🎨 Creative and Research Tasks

### Research Assistant
```
# Ask for explanations
"How does photosynthesis work at the molecular level?"

# Request comparisons
"Compare functional programming vs object-oriented programming"

# Get summaries
"Summarize the key points from the latest IPCC climate report"
```

### Creative Writing
```
# Generate ideas
"Give me 10 creative story ideas about time travel"

# Improve writing
"Rewrite this paragraph to be more engaging: [paste text]"

# Brainstorm solutions
"I'm stuck on this design problem. Here are the constraints: [describe]. What are some creative solutions?"
```

### Learning and Education
```
# Explain concepts
"Explain quantum entanglement like I'm 5"

# Create study plans
"Create a 3-month study plan for learning React.js"

# Practice problems
"Give me 5 practice problems for learning SQL joins"
```

## ⚙️ Customization and Preferences

### Setting Your Preferences
```bash
# Tell Brein AI your preferences
"I prefer concise answers over verbose explanations"
"I work in the healthcare industry, so consider medical context"
"I speak British English, not American English"
"I want responses in Spanish, not English"
```

### Managing Conversation Sessions
```bash
# Start a new session
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "session_id": "new_session_123"}'

# Continue existing session
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Continue our discussion", "session_id": "existing_session_123"}'
```

## 🔒 Privacy and Security

### What Gets Stored
- **Your Messages**: Stored for conversation continuity
- **AI Responses**: Kept for learning and improvement
- **Learned Facts**: Information you explicitly teach the system
- **Web Content**: Only when you enable web access (with safety filtering)

### Your Privacy Controls
```bash
# Disable web access globally
echo '{"security": {"web_access_default": false}}' > config.json

# Clear your conversation history
curl -X POST http://localhost:8000/api/admin/clear-history

# View what the system knows about you
curl http://localhost:8000/api/memory/search?q=about_me
```

### Data Export
```bash
# Export your data
curl -X GET http://localhost:8000/api/user/export-data > my_data.json

# Request data deletion
curl -X POST http://localhost:8000/api/user/delete-data \
  -H "Content-Type: application/json" \
  -d '{"confirmation": "I understand this will delete all my data"}'
```

## 📱 Mobile and Offline Usage

### Mobile Apps (Future)
When mobile apps are available:
- **Offline Mode**: Access learned information without internet
- **Sync**: Seamless synchronization across devices
- **Voice Interface**: Hands-free interaction
- **Push Notifications**: Important updates and reminders

### Current Mobile Access
```bash
# Access via mobile browser
# Replace localhost with your server IP
curl -X POST http://YOUR_SERVER_IP:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello from mobile"}'
```

## 🛠️ Advanced Features

### API Integration
```python
import requests

class BreinAI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def query(self, text, **kwargs):
        response = requests.post(
            f"{self.base_url}/api/query",
            json={"query": text, **kwargs}
        )
        return response.json()

    def teach(self, fact):
        return self.query(f"Remember that {fact}")

    def recall(self, topic):
        return self.query(f"What do you know about {topic}")

# Usage
ai = BreinAI()
ai.teach("I love hiking in the mountains")
response = ai.recall("my hobbies")
```

### Batch Processing
```bash
# Process multiple queries
curl -X POST http://localhost:8000/api/batch-query \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is machine learning?",
      "Explain neural networks",
      "How do they differ?"
    ],
    "session_id": "batch_session_123"
  }'
```

## 📊 Monitoring and Analytics

### System Health
```bash
# Check system status
curl http://localhost:8000/health

# View performance metrics
curl http://localhost:8000/api/system/metrics
```

### Usage Analytics
```bash
# View your usage statistics
curl http://localhost:8000/api/user/analytics

# Conversation history
curl http://localhost:8000/api/user/conversations
```

## 🚨 Troubleshooting Common Issues

### Slow Responses
- **Check internet connection** for web-enabled queries
- **Clear cache** if memory usage is high
- **Restart system** if performance degrades
- **Check system resources** (CPU, RAM)

### Inconsistent Responses
- **Provide more context** in your questions
- **Use specific session IDs** for conversation continuity
- **Teach preferences** explicitly to the system

### Memory Issues
- **Be specific** when teaching information
- **Use clear categories** for organizing knowledge
- **Regular cleanup** of outdated information

## 🎯 Best Practices

### Effective Communication
1. **Be Clear and Specific**: Vague questions get vague answers
2. **Provide Context**: Include relevant background information
3. **Ask Follow-ups**: Build on previous conversations
4. **Give Feedback**: Tell the system what works and what doesn't

### Memory Management
1. **Teach Gradually**: Add information in manageable chunks
2. **Use Categories**: Organize information by topic
3. **Review Regularly**: Check what the system has learned
4. **Correct Mistakes**: Update incorrect information promptly

### Privacy Awareness
1. **Know What Gets Stored**: Review privacy settings
2. **Use Appropriate Sessions**: Separate work and personal conversations
3. **Regular Audits**: Check stored information periodically
4. **Secure Access**: Use strong authentication when available

## 🔄 Updates and Improvements

### Staying Current
- **Check for Updates**: Regular system updates improve performance
- **Backup Important Data**: Before major updates
- **Review Changelog**: Understand new features and changes
- **Update Preferences**: As the system learns your patterns

### Feature Requests
```bash
# Submit feedback
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "type": "feature_request",
    "title": "Add voice input support",
    "description": "It would be great to have voice input capabilities"
  }'
```

## 📚 Learning Resources

### Built-in Help
```bash
# Ask for help
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I use advanced search features?"}'
```

### Documentation Links
- [[Quick Start|Quick-Start]] - Get started quickly
- [[API Reference|API-Reference]] - Technical API details
- [[Troubleshooting|Troubleshooting]] - Solve common problems
- [[Configuration|Configuration]] - Customize your experience

## 🎉 Advanced Usage Examples

### Research Workflow
```
1. "Find recent papers on transformer architectures"
2. "Summarize the key findings from these papers"
3. "Compare the approaches used in each paper"
4. "What are the practical implications for my project?"
```

### Creative Collaboration
```
1. "Brainstorm 5 names for a new mobile app about fitness"
2. "For the best name, create a logo description"
3. "Write a tagline for this app"
4. "Design a user onboarding flow"
```

### Learning Assistant
```
1. "Create a study plan for learning data structures"
2. "Explain binary trees with examples"
3. "Give me practice problems on tree traversal"
4. "Check my solution and explain any mistakes"
```

## 📞 Support and Community

### Getting Help
- **In-System Help**: Ask Brein AI about its own features
- **Documentation**: Comprehensive wiki documentation
- **Community**: GitHub discussions and issues
- **Direct Support**: abdijabarboxer2009@gmail.com

### Contributing Feedback
Your feedback helps improve Brein AI:
- **Bug Reports**: Use GitHub issues
- **Feature Requests**: GitHub discussions
- **User Experience**: Share your usage patterns
- **Performance Feedback**: Report speed and quality issues

---

*User Manual - Last updated: November 2025*
