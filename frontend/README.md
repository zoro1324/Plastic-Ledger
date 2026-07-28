# PlasticLedger AI - Frontend

Premium, production-ready frontend for an AI-powered environmental intelligence platform.

## 🌊 Project Overview

PlasticLedger AI detects floating plastic pollution in satellite imagery, classifies polymer types, estimates degradation, and traces pollution back to its source using advanced AI and geospatial analysis.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ (recommended: 18+)
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The application will be available at `http://localhost:5173`

## 📦 Tech Stack

### Core Framework
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first CSS
- **Framer Motion** - Animations and interactions

### Maps & Geospatial
- **React Leaflet** - Interactive mapping
- **Leaflet Draw** - Drawing tools
- **Turf.js** - Geospatial analysis
- **Leaflet GeoJSON** - Vector data

### Data Visualization
- **Recharts** - Data charts and graphs
- **Three.js** - 3D graphics (optional)
- **React Three Fiber** - Three.js integration

### UI Components
- **ShadCN UI** - Component library
- **Lucide React** - Icons
- **Radix UI** - Headless components

### State Management & Data
- **React Query** - Server state management
- **Context API** - Local state
- **Axios** - HTTP client
- **React Hook Form** - Form management

### Routing
- **React Router v6** - Client-side routing

## 📁 Project Structure

```
src/
├── components/
│   ├── layout/              # Navigation, Footer, Layout
│   ├── dashboard/           # KPI cards, AI workflow
│   ├── gis/                 # Map, toolbar, popups
│   └── ui/                  # Reusable UI components
├── pages/
│   ├── LandingPage.tsx      # Hero landing page
│   ├── DashboardPage.tsx    # Main GIS dashboard
│   ├── AnalyticsPage.tsx    # Analytics dashboard
│   ├── ReportsPage.tsx      # Reports management
│   └── AboutPage.tsx        # About & information
├── context/
│   └── DashboardContext.tsx # Global state management
├── hooks/
│   ├── useAnimations.ts     # Animation hooks
│   ├── useGeoLocation.ts    # Geolocation
│   └── ...other custom hooks
├── lib/
│   ├── api/
│   │   └── mockAPI.ts       # Mock API endpoints
│   └── utils/
│       └── geoUtils.ts      # Geospatial utilities
├── types/
│   └── index.ts             # TypeScript interfaces
├── App.tsx                  # Main app component
├── main.tsx                 # Entry point
└── index.css                # Global styles
```

## 🎨 Design System

### Color Palette
```
Primary Background:  #071A2E
Secondary:          #0F2D4A
Primary Accent:     #0A84FF (Ocean Blue)
Ocean Accent:       #00C2A8 (Teal)
Highlight:          #4CC9F0 (Light Blue)
Success:            #22C55E
Warning:            #F59E0B
Danger:             #EF4444
Text:               White / Light Gray
```

### Core Styles
- Dark ocean theme with glassmorphism
- Rounded cards with soft shadows
- Animated gradients
- Minimalistic layout
- Scientific visualization approach

## 🗺️ Key Features

### Landing Page
- Animated hero section with gradient text
- Feature cards with hover effects
- Statistics section
- Call-to-action buttons
- Responsive design

### Dashboard (Main Feature)
- **Interactive GIS Map**: Full-screen Leaflet map with multiple layers
- **GIS Toolbar**: Drawing tools (rectangle, polygon, markers)
- **Region Selection**: Calculate area, perimeter, dimensions
- **KPI Cards**: Live statistics with animations
- **Detection Results**: Display plastic cluster hotspots
- **Cluster Popups**: Detailed information per cluster
- **AI Workflow**: Animated pipeline progress

### Analytics Page
- Monthly pollution trends
- Polymer distribution charts
- Risk region rankings
- Detection accuracy over time
- Industrial source analysis
- Cleanup progress tracking

### Reports Page
- Report cards with export options
- PDF, CSV, GeoJSON downloads
- Share and print functionality
- Report management

### About Page
- Mission and vision statements
- Technology stack details
- SDG alignment
- Research methodology
- Contributing partners

## 🔧 Configuration

### Environment Variables
Create a `.env.local` file:

```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_MAPBOX_TOKEN=your_token_here
VITE_APP_TITLE=PlasticLedger AI
```

### Tailwind Configuration
Customize in `tailwind.config.ts`:
- Extend color palette
- Add custom utilities
- Configure responsive breakpoints

## 🎬 Animations

The app uses Framer Motion for:
- Page transitions
- Component entrance/exit
- Hover effects
- Loading states
- Progress animations
- Scroll animations

## 📊 Mock API

The application includes comprehensive mock APIs in `src/lib/api/mockAPI.ts`:

- `selectRegionAPI()` - Region selection
- `calculateAreaAPI()` - Geospatial calculations
- `plasticsDetectionAPI()` - Detection pipeline
- `polymerAnalysisAPI()` - Polymer classification
- `sourceAttributionAPI()` - Source tracing
- `dashboardStatsAPI()` - Statistics
- `analyticsAPI()` - Analytics data
- `exportReportAPI()` - Report export

## 🧪 Testing

```bash
# Run tests
npm run test

# Watch mode
npm run test:watch

# Run Playwright tests
npm run test:e2e
```

## 🔍 Code Quality

```bash
# Run ESLint
npm run lint

# Format with Prettier (if configured)
npm run format
```

## 📈 Performance Optimization

- Code splitting via Vite
- Image optimization
- CSS-in-JS minimization
- Tree shaking
- Lazy component loading
- Memoization of expensive components

## 🌐 Responsiveness

The application is fully responsive:
- **Mobile**: 320px+
- **Tablet**: 768px+
- **Desktop**: 1024px+
- **Large Screens**: 1440px+

## 🚢 Deployment

### Build
```bash
npm run build
```

Output goes to `dist/` directory.

### Deploy to Vercel
```bash
npm i -g vercel
vercel
```

### Deploy to Netlify
```bash
npm i -g netlify-cli
netlify deploy --prod --dir=dist
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

## 📚 Available Scripts

```bash
npm run dev              # Start dev server
npm run build            # Build for production
npm run build:dev        # Build in dev mode
npm run preview          # Preview production build
npm run lint             # Run ESLint
npm run test             # Run tests
npm run test:watch       # Run tests in watch mode
```

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit changes (`git commit -m 'Add amazing feature'`)
3. Push to branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📝 Component Guidelines

### Creating New Components

```typescript
import React from 'react';
import { motion } from 'framer-motion';

interface MyComponentProps {
  title: string;
  onAction?: () => void;
}

export const MyComponent: React.FC<MyComponentProps> = ({ title, onAction }) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {title}
    </motion.div>
  );
};
```

### Using Hooks

```typescript
import { useAnimatedCounter, useLocalStorage } from '@/hooks/useAnimations';

const [value] = useAnimatedCounter(100, 1000);
const [savedData, setSavedData] = useLocalStorage('key', defaultValue);
```

## 🐛 Troubleshooting

### Map not loading
- Check Leaflet CSS is imported
- Verify map container has height
- Check browser console for errors

### Animations not smooth
- Ensure GPU acceleration is enabled
- Check Framer Motion configuration
- Verify performance in DevTools

### API calls failing
- Check mock API implementation
- Verify environment variables
- Check network tab in DevTools

## 📖 Additional Resources

- [Vite Documentation](https://vitejs.dev)
- [React Documentation](https://react.dev)
- [TailwindCSS](https://tailwindcss.com)
- [Framer Motion](https://www.framer.com/motion)
- [React Leaflet](https://react-leaflet.js.org)
- [Recharts](https://recharts.org)
- [TypeScript](https://www.typescriptlang.org)

## 📄 License

This project is for research and educational purposes.
