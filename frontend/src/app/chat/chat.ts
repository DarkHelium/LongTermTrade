import { Component, signal, ElementRef, ViewChild, AfterViewChecked, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  suggestions?: { symbol: string; qty: number; term: string }[];
  timestamp?: Date;
  reasoning_traces?: string[];
  tool_usage?: Array<{tool: string; input: string; result: string}>;
  agent_context?: string;
  iterations?: number;
}

interface ChatResponse {
  response: string;
  buffett_picks?: string[];
  timestamp: Date;
  reasoning_traces?: string[];
  tool_usage?: Array<{tool: string; input: string; result: string}>;
  agent_context?: string;
}

interface AgentChatResponse {
  response: string;
  reasoning_traces: string[];
  tool_usage: Array<{tool: string; input: string; result: string}>;
  iterations: number;
  context: string;
  buffett_picks?: string[];
  timestamp: Date;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrl: './chat.css'
})
export class Chat implements AfterViewChecked, OnInit {
  @ViewChild('chatMessages') private chatMessagesContainer!: ElementRef;
  @ViewChild('messageInput') private messageInput!: ElementRef;
  
  messages = signal<Message[]>([]);
  newMessage = '';
  isLive = false;
  useAdvancedAgent = signal(false);
  showAgentDetails = signal(false);
  private shouldScrollToBottom = false;
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) {}

  ngOnInit() {
    // Initialize with Warren Buffett's welcome message and top picks
    this.initializeWithTopPicks();
  }

  ngAfterViewChecked() {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
  }

  async initializeWithTopPicks() {
    // Show initial loading message
    const loadingMessage: Message = {
      role: 'assistant',
      content: `Hello! I'm Warren Buffett's AI assistant. Let me get my current top stock recommendations for you...`,
      timestamp: new Date()
    };
    
    this.messages.set([loadingMessage]);
    this.shouldScrollToBottom = true;

    try {
      // Get dynamic picks from the AI agent
      const response = await this.http.post<AgentChatResponse>(`${this.apiUrl}/agent-chat`, {
        message: "What are your current top 5 stock picks based on Warren Buffett's investment criteria? Please provide specific stock symbols and brief reasons for each recommendation.",
        use_agent: true
      }).toPromise();

      if (response) {
        const welcomeMessage: Message = {
          role: 'assistant',
          content: `Hello! I'm Warren Buffett's AI assistant. ${response.response}

What would you like to know about any of these stocks or my investment approach?`,
          timestamp: new Date(),
          reasoning_traces: response.reasoning_traces || [],
          tool_usage: response.tool_usage || [],
          agent_context: response.context,
          iterations: response.iterations
        };
        
        // Add buffett picks if available
        if (response.buffett_picks && response.buffett_picks.length > 0) {
          welcomeMessage.suggestions = response.buffett_picks.map(symbol => ({
            symbol,
            qty: 10,
            term: 'long'
          }));
        }
        
        this.messages.set([welcomeMessage]);
      } else {
        throw new Error('No response from AI agent');
      }
    } catch (error) {
      console.error('Error getting AI picks:', error);
      // Fallback to a simple welcome message if AI fails
      const fallbackMessage: Message = {
        role: 'assistant',
        content: `Hello! I'm Warren Buffett's AI assistant. I can help you understand my investment philosophy and analyze stocks using my criteria.

I'm currently having trouble accessing my latest stock recommendations, but I'm ready to help you analyze any specific stocks you're interested in.

What would you like to know about my investment approach or any particular stock?`,
        timestamp: new Date()
      };
      
      this.messages.set([fallbackMessage]);
    }
    
    this.shouldScrollToBottom = true;
  }

  sendMessage() {
    if (!this.newMessage.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: this.newMessage.trim(),
      timestamp: new Date()
    };

    this.messages.update(msgs => [...msgs, userMessage]);
    const query = this.newMessage.trim();
    this.newMessage = '';
    this.shouldScrollToBottom = true;

    // Focus back to input after sending
    setTimeout(() => {
      if (this.messageInput) {
        this.messageInput.nativeElement.focus();
      }
    }, 100);

    // Choose endpoint based on agent mode
    const endpoint = this.useAdvancedAgent() ? '/agent-chat' : '/chat';
    const requestBody = this.useAdvancedAgent() 
      ? { message: query, use_agent: true }
      : { message: query };

    this.http.post<ChatResponse | AgentChatResponse>(`${this.apiUrl}${endpoint}`, requestBody)
      .subscribe({
        next: (data) => {
          const assistantMessage: Message = {
            role: 'assistant',
            content: data.response,
            timestamp: new Date(),
            reasoning_traces: data.reasoning_traces || [],
            tool_usage: data.tool_usage || [],
            agent_context: 'context' in data ? data.context : undefined,
            iterations: 'iterations' in data ? data.iterations : undefined
          };
          
          // Add buffett picks if available
          if (data.buffett_picks && data.buffett_picks.length > 0) {
            assistantMessage.suggestions = data.buffett_picks.map(symbol => ({
              symbol,
              qty: 10,
              term: 'long'
            }));
          }
          
          this.messages.update(msgs => [...msgs, assistantMessage]);
          this.shouldScrollToBottom = true;
        },
        error: (error) => {
          console.error('Error sending message:', error);
          let errorMsg = 'Sorry, I encountered an error. Please try again.';
          if (error && error.error && typeof error.error === 'string') {
            errorMsg += `\nDetails: ${error.error}`;
          } else if (error && error.message) {
            errorMsg += `\nDetails: ${error.message}`;
          } else if (error && error.status) {
            errorMsg += `\nStatus: ${error.status}`;
          }
          const errorMessage: Message = {
            role: 'assistant',
            content: errorMsg,
            timestamp: new Date()
          };
          this.messages.update(msgs => [...msgs, errorMessage]);
          this.shouldScrollToBottom = true;
        }
      });
  }

  invest(suggestions: { symbol: string; qty: number; term: string }[]) {
    this.http.post('http://localhost:3000/invest', suggestions)
      .subscribe({
        next: (response) => {
          const successMessage = `✅ Investments executed successfully: ${JSON.stringify(response)}`;
          const confirmationMessage: Message = {
            role: 'assistant',
            content: successMessage,
            timestamp: new Date()
          };
          this.messages.update(msgs => [...msgs, confirmationMessage]);
          this.shouldScrollToBottom = true;
        },
        error: (error) => {
          console.error('Error executing investments:', error);
          const errorMessage: Message = {
            role: 'assistant',
            content: '❌ Failed to execute investments. Please try again or contact support.',
            timestamp: new Date()
          };
          this.messages.update(msgs => [...msgs, errorMessage]);
          this.shouldScrollToBottom = true;
        }
      });
  }

  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  onKeyPress(event: KeyboardEvent) {
    // Additional key press handling if needed
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
    }
  }

  getCurrentTime(): string {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  toggleAgentMode() {
    this.useAdvancedAgent.update(current => !current);
  }

  toggleAgentDetails() {
    this.showAgentDetails.update(current => !current);
  }

  formatToolUsage(toolUsage: Array<{tool: string; input: string; result: string}>): string {
    return toolUsage.map(tool => 
      `🔧 ${tool.tool}: ${tool.input} → ${tool.result.substring(0, 50)}...`
    ).join('\n');
  }

  private scrollToBottom(): void {
    try {
      if (this.chatMessagesContainer) {
        const element = this.chatMessagesContainer.nativeElement;
        element.scrollTop = element.scrollHeight;
      }
    } catch (err) {
      console.error('Error scrolling to bottom:', err);
    }
  }

  parseMarkdown(content: string): SafeHtml {
    const html = marked.parse(content);
    return this.sanitizer.bypassSecurityTrustHtml(html as string);
  }
}
