# Frontend Redesign - Design Document

## Overview

This document outlines the technical design for redesigning the Medicine Assistant frontend into a professional, accessible, and user-friendly healthcare application.

### Feature Separation Architecture

The application follows a clear separation between pre-login and post-login features:

**Pre-Login (Landing Page)**:
- Emergency Drug Check: Quick, anonymous interaction checking without account creation
- Feature showcase and marketing content
- Login/Register CTAs

**Post-Login (Dashboard)**:
- Add New Medication: Personalized medication addition with profile integration
- Medication Wallet: Saved medications management
- Profile-based interaction checking

**Rationale**: Emergency Drug Check is designed for quick, anonymous use and would create functional duplication if shown after login. Live Health Monitoring (Google Fit) was a standalone demo feature not directly connected to core medication interaction, dosage validation, or medication storage workflows. The dashboard focuses exclusively on authenticated medication management features.

## Architecture

### Component Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Navbar.jsx (redesigned)
│   │   ├── Footer.jsx (new)
│   │   └── Layout.jsx (updated)
│   ├── landing/
│   │   ├── Hero.jsx (new)
│   │   ├── QuickCheckModal.jsx (new)
│   │   └── Features.jsx (new)
│   ├── medication/
│   │   ├── MedicationChip.jsx (new)
│   │   ├── DrugSearch.jsx (new)
│   │   ├── DosageInput.jsx (new)
│   │   └── MedicationWallet.jsx (redesigned)
│   ├── risk/
│   │   ├── RiskGauge.jsx (new)
│   │   ├── RiskBadge.jsx (new)
│   │   └── ExplainPanel.jsx (new)
│   └── common/
│       ├── Button.jsx (new)
│       ├── Card.jsx (new)
│       ├── Toast.jsx (new)
│       └── Modal.jsx (new)
├── pages/
│   ├── Landing.jsx (new)
│   │   - Emergency Drug Check
│   │   - Feature showcase
│   │   - Login/Register CTAs
│   ├── Dashboard.jsx (redesigned)
│   │   - Add New Medication form
│   │   - Medication Wallet
│   │   - NO Emergency Check (pre-login only)
│   │   - NO Live Health Monitoring (removed)
│   ├── Profile.jsx (redesigned)
│   └── History.jsx (new)
└── styles/
    └── index.css (updated with design system)
```

## Components and Interfaces

### 1. Design System Components

#### Button Component
```jsx
<Button 
  variant="primary|secondary|danger"
  size="sm|md|lg"
  icon={<Icon />}
  loading={boolean}
>
  Text
</Button>
```

#### Card Component
```jsx
<Card 
  shadow="soft|soft-lg"
  padding="sm|md|lg"
  rounded="card|card-lg"
>
  Content
</Card>
```

### 2. Landing Page Components

#### Hero Section
- Headline: "Is that medicine safe for you?"
- Subheadline: "Quickly check interactions — no signup required"
- Primary CTA: "Check Interaction"
- Secondary link: "Check with your profile (more accurate)"

#### QuickCheckModal
- Drug search with autocomplete
- Add multiple drugs as chips
- "Check Risk" button
- Optional: "Check with my profile" checkbox

### 3. Risk Analysis Components

#### RiskGauge
- Circular progress indicator (0-100)
- Color-coded: green (<30), yellow (30-70), red (>70)
- Animated on load
- Center text shows percentage

#### RiskBadge
- Pill-shaped badge
- Text: SAFE / CAUTION / DANGEROUS
- Color-coded background
- Icon indicator

#### ExplainPanel
- Collapsible panel
- Plain language explanation
- Technical details (optional tab)
- Evidence sources
- Suggested alternatives

### 4. Medication Components

#### MedicationChip
- Pill icon
- Drug name
- Dosage (small text)
- Remove button (X)
- Hover effects

#### DrugSearch
- Autocomplete input
- Dropdown with suggestions
- Brand name hints
- Loading state

#### MedicationWallet
- Grid of medication cards
- Each card shows:
  - Drug name and icon
  - Dosage and frequency
  - Next dose countdown
  - Edit/Delete buttons
  - Reminder toggle

## Data Models

### Risk Analysis Result
```typescript
interface RiskAnalysisResult {
  gnn_risk: number;           // 0-100
  verdict: string;            // "SAFE TO ADD" | "DO NOT ADD"
  ai_response: string;        // Plain text explanation
  can_add: boolean;
  dosage_validation: {
    is_safe: boolean;
    warnings: string[];
    max_daily: number | string;
    max_single: number | string;
  };
}
```

### Medication
```typescript
interface Medication {
  id: number;
  drug_name: string;
  dosage_amount: number;
  dosage_unit: string;
  frequency: string;
  start_date: string;
  end_date?: string;
}
```

## Error Handling

1. **API Errors**: Show toast notification with retry option
2. **Validation Errors**: Inline error messages below inputs
3. **Network Errors**: Friendly message with offline indicator
4. **Rate Limit Errors**: "Please wait a moment" message

## Testing Strategy

1. **Component Testing**: Test each component in isolation
2. **Integration Testing**: Test user flows end-to-end
3. **Accessibility Testing**: Use Lighthouse and axe-core
4. **Responsive Testing**: Test on mobile, tablet, desktop
5. **Browser Testing**: Chrome, Firefox, Safari, Edge

## Performance Considerations

1. **Code Splitting**: Lazy load pages and heavy components
2. **Image Optimization**: Use WebP format, lazy loading
3. **Animation Performance**: Use CSS transforms, avoid layout thrashing
4. **Bundle Size**: Keep under 500KB gzipped
5. **Lighthouse Score**: Target 90+ for all metrics
