import React from 'react';
import { LogRows, CustomScrollbar } from '@grafana/ui';
import { LogsDedupStrategy, PanelProps } from '@grafana/data';
import { Options } from './types';
import { dataFrameToLogsModel } from 'DeepWheels/GrafanaServer/public/app/core/logs_model';
import { sortLogsResult } from 'DeepWheels/GrafanaServer/public/app/core/utils/explore';

interface LogsPanelProps extends PanelProps<Options> {}

export const LogsPanel: React.FunctionComponent<LogsPanelProps> = ({
  data,
  timeZone,
  options: { showLabels, showTime, wrapLogMessage, sortOrder },
  width,
}) => {
  if (!data) {
    return (
      <div className="panel-empty">
        <p>No data found in response</p>
      </div>
    );
  }

  const newResults = data ? dataFrameToLogsModel(data.series, data.request?.intervalMs, timeZone) : null;
  const sortedNewResults = sortLogsResult(newResults, sortOrder);

  return (
    <CustomScrollbar autoHide>
      <LogRows
        logRows={sortedNewResults.rows}
        dedupStrategy={LogsDedupStrategy.none}
        highlighterExpressions={[]}
        showLabels={showLabels}
        showTime={showTime}
        wrapLogMessage={wrapLogMessage}
        timeZone={timeZone}
        allowDetails={true}
        disableCustomHorizontalScroll={true}
      />
    </CustomScrollbar>
  );
};
