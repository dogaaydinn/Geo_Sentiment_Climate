# Geo Climate Mobile App

Cross-platform mobile app for air quality predictions using React Native.

## Features

- 📱 Cross-platform (iOS & Android)
- 🔮 Real-time air quality predictions
- 📊 Historical data visualization
- 🗺️ Location-based forecasts
- 🔔 Push notifications for air quality alerts
- 📴 Offline mode with local caching
- 📍 GPS integration for current location

## Prerequisites

- Node.js >= 16
- React Native development environment setup
- iOS: Xcode 14+ (Mac only)
- Android: Android Studio with SDK 33+

## Installation

```bash
# Install dependencies
cd mobile/geo-climate-app
npm install

# iOS only - Install pods
cd ios && pod install && cd ..
```

## Running

```bash
# Start Metro bundler
npm start

# Run on Android
npm run android

# Run on iOS
npm run ios
```

## Project Structure

```
src/
├── screens/          # Screen components
│   ├── HomeScreen.tsx
│   ├── PredictionScreen.tsx
│   └── SettingsScreen.tsx
├── components/       # Reusable components
│   ├── AQICard.tsx
│   ├── PredictionChart.tsx
│   └── LocationPicker.tsx
├── services/         # API and data services
│   ├── api.ts
│   ├── location.ts
│   └── notifications.ts
├── utils/           # Utility functions
│   ├── formatters.ts
│   └── validators.ts
└── types/           # TypeScript types
    └── index.ts
```

## Configuration

### API Key

Set your API key in the app settings or configure it via environment:

```bash
echo "GEO_CLIMATE_API_KEY=your-api-key" > .env
```

### Push Notifications

Configure Firebase Cloud Messaging (Android) and APNs (iOS) for push notifications.

## Building for Production

### Android

```bash
cd android
./gradlew assembleRelease
```

### iOS

```bash
cd ios
xcodebuild -workspace GeoClimate.xcworkspace \
           -scheme GeoClimate \
           -configuration Release
```

## Features Roadmap

- [x] Basic prediction interface
- [x] SDK integration
- [ ] Offline mode
- [ ] Push notifications
- [ ] Location services
- [ ] Historical charts
- [ ] Widget support
- [ ] Watch app integration

## License

Apache 2.0

## Support

- **Issues**: https://github.com/dogaaydinn/Geo_Sentiment_Climate/issues
- **Docs**: https://docs.geo-climate.com
