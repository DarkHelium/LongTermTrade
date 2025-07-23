import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./dashboard/dashboard').then(m => m.Dashboard),
  },
  {
    path: 'chat',
    loadComponent: () => import('./chat/chat').then(m => m.Chat),
  },
];
