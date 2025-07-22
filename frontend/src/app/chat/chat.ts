import { Component, signal, ElementRef, ViewChild, AfterViewChecked } from '@angular/core';
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
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrl: './chat.css'
})
export class Chat implements AfterViewChecked {
  @ViewChild('chatMessages') private chatMessagesContainer!: ElementRef;
  @ViewChild('messageInput') private messageInput!: ElementRef;
  
  messages = signal<Message[]>([]);
  newMessage = '';
  isLive = false;
  private shouldScrollToBottom = false;

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) {}

  ngAfterViewChecked() {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
      this.shouldScrollToBottom = false;
    }
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

    this.http.post<{ response: string; suggestions: { symbol: string; qty: number; term: string }[] }>('http://localhost:3000/chat', { query, is_live: this.isLive })
      .subscribe({
        next: (data) => {
          const assistantMessage: Message = {
            role: 'assistant',
            content: data.response,
            suggestions: data.suggestions,
            timestamp: new Date()
          };
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
