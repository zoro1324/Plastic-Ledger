# PlasticLedger AI - Development Guide

## Getting Started

### Prerequisites
- Node.js 16+ (recommended: 18+)
- npm 7+ or yarn
- Git

### Initial Setup

```bash
# Clone the repository
git clone <repo-url>
cd Plastic-Ledger/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:5173 in your browser
```

## Project Architecture

### Core Layers

```
Presentation Layer (Pages & Components)
        ↓
Business Logic Layer (Hooks & Context)
        ↓
Data Layer (APIs & State Management)
        ↓
Utilities Layer (Helpers & Formatters)
```

### Component Hierarchy

```
App
├── DashboardProvider (Context)
├── Router
│   ├── LandingPage
│   ├── DashboardPage
│   │   ├── Navigation
│   │   ├── KPIGrid
│   │   ├── GISMap
│   │   ├── GISToolbar
│   │   ├── AIWorkflow
│   │   └── Panels
│   ├── AnalyticsPage
│   ├── ReportsPage
│   └── AboutPage
└── Footer
```

## Development Workflow

### 1. Creating a New Feature

#### Step 1: Create the Type
If needed, add TypeScript interfaces in `src/types/index.ts`:

```typescript
export interface NewFeature {
  id: string;
  name: string;
  // ... properties
}
```

#### Step 2: Create the Component
Create component in `src/components/` with proper structure:

```typescript
import React from 'react';
import { motion } from 'framer-motion';

interface NewComponentProps {
  title: string;
  onAction?: () => void;
}

export const NewComponent: React.FC<NewComponentProps> = ({ title, onAction }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {title}
    </motion.div>
  );
};
```

#### Step 3: Add Mock API (if needed)
Update `src/lib/api/mockAPI.ts`:

```typescript
export const newFeatureAPI = async (): Promise<NewFeature[]> => {
  await SIMULATED_DELAY(500);
  return [/* mock data */];
};
```

#### Step 4: Integrate with Page
Import and use the component in your page component.

### 2. Working with State Management

#### Using Context
```typescript
import { useDashboard } from '@/context/DashboardContext';

function MyComponent() {
  const { selectedRegion, setSelectedRegion } = useDashboard();
  // Use state...
}
```

#### Using Custom Hooks
```typescript
import { useAnimatedCounter, useLocalStorage } from '@/hooks/useAnimations';

function MyComponent() {
  const animatedValue = useAnimatedCounter(100, 1000);
  const [saved, setSaved] = useLocalStorage('key', defaultValue);
}
```

### 3. Adding Animations

#### Framer Motion Basics
```typescript
import { motion } from 'framer-motion';

<motion.div
  initial={{ opacity: 0, scale: 0.9 }}
  animate={{ opacity: 1, scale: 1 }}
  exit={{ opacity: 0, scale: 0.9 }}
  transition={{ duration: 0.5 }}
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
>
  Content
</motion.div>
```

#### Container Animations
```typescript
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

<motion.div variants={containerVariants} initial="hidden" animate="visible">
  {items.map((item) => (
    <motion.div key={item.id} variants={itemVariants}>
      {item.content}
    </motion.div>
  ))}
</motion.div>
```

### 4. Styling with TailwindCSS

#### Common Patterns
```typescript
// Gradient backgrounds
className="bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0]"

// Glassmorphism
className="backdrop-blur-md bg-white/10 border border-white/20"

// Dark theme colors
className="bg-[#071A2E] text-white"

// Responsive
className="px-4 md:px-6 lg:px-8"

// Transitions
className="transition-all duration-300 hover:scale-105"
```

### 5. Working with Maps

#### Adding a Marker
```typescript
import { CircleMarker } from 'react-leaflet';

<CircleMarker
  center={[latitude, longitude]}
  radius={size}
  fillColor={color}
  color={strokeColor}
  fillOpacity={opacity}
  eventHandlers={{
    click: () => handleClick(),
  }}
/>
```

#### Calculating Distances
```typescript
import { calculateDistance, calculateBearing } from '@/lib/utils/geoUtils';

const distance = calculateDistance(from, to); // in km
const bearing = calculateBearing(from, to); // in degrees
```

### 6. Working with Charts

#### Simple Line Chart
```typescript
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

<ResponsiveContainer width="100%" height={300}>
  <LineChart data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="name" />
    <YAxis />
    <Tooltip />
    <Legend />
    <Line type="monotone" dataKey="value" stroke="#0A84FF" />
  </LineChart>
</ResponsiveContainer>
```

## File Organization Best Practices

### Component Files
```
src/components/[feature]/
├── Component.tsx       # Main component
├── Component.test.tsx  # Tests
├── Component.styles.ts # Styled components (if needed)
└── types.ts           # Local types
```

### Page Files
```
src/pages/
├── FeaturePage.tsx    # Page component
└── FeaturePage.test.tsx # Tests
```

### API Files
```
src/lib/api/
├── mockAPI.ts         # Mock endpoints
├── endpoints.ts       # Real endpoints (when API is ready)
└── types.ts          # API response types
```

## Common Tasks

### Adding a New Page
1. Create `src/pages/NewPage.tsx`
2. Add route in `src/App.tsx`
3. Add navigation link in `src/components/layout/Navigation.tsx`

### Adding a New Chart
1. Import from Recharts
2. Format data in the page component
3. Wrap in ResponsiveContainer for responsiveness
4. Add tooltips and legends

### Adding Form Validation
```typescript
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const schema = z.object({
  email: z.string().email(),
  name: z.string().min(1),
});

const { register, handleSubmit, errors } = useForm({
  resolver: zodResolver(schema),
});
```

### Handling Async Operations
```typescript
const { data, loading, error } = useFetchData(
  () => myAPI(),
  [dependencies]
);

if (loading) return <LoadingSpinner />;
if (error) return <ErrorMessage error={error} />;
return <Component data={data} />;
```

## Testing

### Unit Tests
```bash
npm run test
```

### E2E Tests
```bash
npm run test:e2e
```

### Test File Structure
```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Component } from './Component';

describe('Component', () => {
  beforeEach(() => {
    // Setup
  });

  it('should render', () => {
    render(<Component />);
    expect(screen.getByText('Expected Text')).toBeInTheDocument();
  });
});
```

## Performance Optimization

### Code Splitting
```typescript
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

<Suspense fallback={<LoadingSpinner />}>
  <HeavyComponent />
</Suspense>
```

### Memoization
```typescript
import { memo, useMemo } from 'react';

const MemoizedComponent = memo(Component);

const memoizedValue = useMemo(() => expensiveCalculation(), [dependencies]);
```

### Debouncing
```typescript
import { useDebounce } from '@/hooks/useAnimations';

const debouncedSearch = useDebounce(searchTerm, 300);
```

## Debugging

### Browser DevTools
1. Open Chrome DevTools (F12)
2. React DevTools extension for component inspection
3. Console for error messages
4. Network tab for API calls

### Console Logging
```typescript
console.log('Value:', value);
console.error('Error:', error);
console.warn('Warning:', warning);
console.table(data); // For tables
```

### Debugging with VS Code
Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Launch Chrome",
      "url": "http://localhost:5173",
      "webRoot": "${workspaceFolder}/src"
    }
  ]
}
```

## Build & Deployment

### Development Build
```bash
npm run build:dev
```

### Production Build
```bash
npm run build
```

### Preview
```bash
npm run preview
```

### Environment Variables
Create `.env.local`:
```
VITE_API_BASE_URL=http://localhost:8000/api
VITE_APP_TITLE=PlasticLedger AI
```

## Troubleshooting

### Hot Module Replacement (HMR) Not Working
- Restart dev server
- Clear browser cache
- Check Vite config in `vite.config.ts`

### Type Errors
- Run `npm install` to ensure dependencies are installed
- Check TypeScript config in `tsconfig.json`
- Use `// @ts-ignore` only as last resort

### Styling Issues
- Check TailwindCSS config
- Verify class names match Tailwind patterns
- Check CSS specificity conflicts

### Build Failures
- Clear `node_modules` and reinstall
- Check for circular dependencies
- Verify all imports are correct

## Git Workflow

### Feature Branch
```bash
git checkout -b feature/my-feature
# Make changes
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

### Pull Request
1. Create PR on GitHub
2. Add description
3. Request review
4. Address feedback
5. Merge when approved

## Resources

- [React Docs](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [TailwindCSS Docs](https://tailwindcss.com/docs)
- [Framer Motion Docs](https://www.framer.com/motion/)
- [Vite Guide](https://vitejs.dev/guide/)
- [React Leaflet](https://react-leaflet.js.org/)
- [Recharts](https://recharts.org/)

## Support

For issues or questions:
1. Check existing documentation
2. Search GitHub issues
3. Create new issue with clear description
4. Include error messages and screenshots
