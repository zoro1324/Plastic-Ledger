import { useEffect, useState, useRef } from 'react';
import { easeOutQuart, easeInOutQuad } from '@/lib/utils/geoUtils';

// Animated counter hook
export const useAnimatedCounter = (end: number, duration: number = 1000, decimals: number = 0) => {
  const [count, setCount] = useState(0);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    if (end === 0) {
      setCount(0);
      return;
    }

    const animate = (timestamp: number) => {
      if (startRef.current === null) {
        startRef.current = timestamp;
      }

      const elapsed = timestamp - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = easeOutQuart(progress);

      setCount(Number((end * eased).toFixed(decimals)));

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        startRef.current = null;
      }
    };

    const frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [end, duration, decimals]);

  return count;
};

// Animated value hook
export const useAnimatedValue = (end: number, duration: number = 500) => {
  const [value, setValue] = useState(0);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    const animate = (timestamp: number) => {
      if (startRef.current === null) {
        startRef.current = timestamp;
      }

      const elapsed = timestamp - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = easeInOutQuad(progress);

      setValue(end * eased);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        startRef.current = null;
      }
    };

    const frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [end, duration]);

  return value;
};

// Fade in animation hook
export const useFadeIn = (duration: number = 500, delay: number = 0) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setIsVisible(true);
    }, delay);

    return () => clearTimeout(timeoutId);
  }, [delay]);

  return {
    ref,
    style: {
      opacity: isVisible ? 1 : 0,
      transition: `opacity ${duration}ms ease-out`,
    },
  };
};

// Scroll animation hook
export const useScrollAnimation = () => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, []);

  return { ref, isVisible };
};

// Pulse animation hook for attention-grabbing elements
export const usePulseAnimation = (isActive: boolean = true) => {
  return {
    animate: isActive ? {
      scale: [1, 1.05, 1],
      opacity: [1, 0.8, 1],
    } : {},
    transition: {
      duration: 2,
      repeat: Infinity,
      repeatType: 'loop' as const,
    },
  };
};

// Progress animation hook
export const useProgressAnimation = (progress: number, duration: number = 500) => {
  const [animatedProgress, setAnimatedProgress] = useState(0);

  useEffect(() => {
    let frameId: number;
    let start: number | null = null;

    const animate = (timestamp: number) => {
      if (!start) start = timestamp;
      const elapsed = timestamp - start;
      const localProgress = Math.min(elapsed / duration, 1);

      setAnimatedProgress(animatedProgress + (progress - animatedProgress) * localProgress);

      if (localProgress < 1) {
        frameId = requestAnimationFrame(animate);
      }
    };

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [progress, duration]);

  return animatedProgress;
};

// Fetch data hook with loading state
export const useFetchData = <T,>(
  fetchFn: () => Promise<T>,
  dependencies: unknown[] = []
) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchFn();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('Unknown error'));
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, dependencies);

  return { data, loading, error };
};

// Debounce hook
export const useDebounce = <T,>(value: T, delay: number = 300): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
};

// Throttle hook
export const useThrottle = <T,>(value: T, interval: number = 300): T => {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastUpdated = useRef<number>(Date.now());

  useEffect(() => {
    const now = Date.now();
    if (now >= lastUpdated.current + interval) {
      lastUpdated.current = now;
      setThrottledValue(value);
    }

    const timerId = setTimeout(
      () => {
        lastUpdated.current = Date.now();
        setThrottledValue(value);
      },
      interval - (now - lastUpdated.current)
    );

    return () => clearTimeout(timerId);
  }, [value, interval]);

  return throttledValue;
};

// Use local storage hook
export const useLocalStorage = <T,>(key: string, initialValue: T): [T, (value: T) => void] => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });

  const setValue = (value: T) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
};
