import config from 'DeepWheels/GrafanaServer/public/app/core/config';

export function getThemeColor(dark: string, light: string): string {
  return config.bootData.user.lightTheme ? light : dark;
}
