import { useState, useEffect } from 'react';

export const useGeoLocation = () => {
  const [latitude, setLatitude] = useState<number | null>(null);
  const [longitude, setLongitude] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLatitude(position.coords.latitude);
          setLongitude(position.coords.longitude);
        },
        (err) => {
          setError(err.message);
          // Default to equator
          setLatitude(0);
          setLongitude(0);
        }
      );
    }
  }, []);

  return { latitude, longitude, error };
};
