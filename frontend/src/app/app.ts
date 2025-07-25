import { Component, OnInit, signal, HostListener } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { Chat } from './chat/chat';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CommonModule, HttpClientModule, FormsModule, Chat],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  protected readonly title = signal('AI Trading Assistant');
  showChat = signal(false);
  
  ngOnInit() {
    // Initialize app
  }

  @HostListener('document:keydown', ['$event'])
  handleKeyboardEvent(event: KeyboardEvent) {
    // Ctrl + U to toggle chat
    if (event.ctrlKey && event.key === 'u') {
      event.preventDefault();
      this.toggleChat();
    }
    // Escape to close chat
    if (event.key === 'Escape' && this.showChat()) {
      this.showChat.set(false);
    }
  }

  toggleChat() {
    this.showChat.update(current => !current);
  }

  closeChat() {
    this.showChat.set(false);
  }
}
