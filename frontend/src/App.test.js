import { render, screen } from '@testing-library/react';
jest.mock('axios', () => ({ post: jest.fn(() => Promise.resolve({ data: {} })) }));
import App from './App';

describe('Dashboard UI', () => {
  test('renders header title', () => {
    render(<App />);
    expect(screen.getByText(/AI Traffic Optimization Dashboard/i)).toBeInTheDocument();
  });

  test('shows upload instructions', () => {
    render(<App />);
    expect(screen.getByText(/Provide Intersection Videos/i)).toBeInTheDocument();
  });
});
