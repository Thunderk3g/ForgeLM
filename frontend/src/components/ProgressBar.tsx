import React from 'react';

interface ProgressBarProps {
  progress: number;
  label?: string;
  className?: string;
  showPercentage?: boolean;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  label,
  className = '',
  showPercentage = true
}) => {
  const clampedProgress = Math.max(0, Math.min(100, progress));

  return (
    <div className={`w-full space-y-2 ${className}`}>
      {label && (
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium leading-none">{label}</span>
          {showPercentage && (
            <span className="text-sm font-medium text-muted-foreground">
              {clampedProgress.toFixed(1)}%
            </span>
          )}
        </div>
      )}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full w-full flex-1 bg-primary transition-all duration-500 ease-out"
          style={{ 
            transform: `translateX(-${100 - clampedProgress}%)`,
            transition: 'transform 0.5s ease-out'
          }}
        />
      </div>
    </div>
  );
};