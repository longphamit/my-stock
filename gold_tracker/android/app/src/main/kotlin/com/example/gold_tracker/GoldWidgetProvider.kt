package com.example.gold_tracker

import android.appwidget.AppWidgetManager
import android.content.Context
import android.content.SharedPreferences
import android.widget.RemoteViews
import es.antonborri.home_widget.HomeWidgetProvider
import es.antonborri.home_widget.HomeWidgetLaunchIntent

class GoldWidgetProvider : HomeWidgetProvider() {
    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray,
        widgetData: SharedPreferences
    ) {
        for (appWidgetId in appWidgetIds) {
            val views = RemoteViews(context.packageName, R.layout.gold_widget).apply {
                // Fetch values set by home_widget Dart side
                val title = widgetData.getString("title", "VÀNG THẾ GIỚI")
                val updatedTime = widgetData.getString("updated_time", "--:--")
                val worldPrice = widgetData.getString("world_price", "-,--- USD")
                val yesterdayPrice = widgetData.getString("yesterday_price", "Hôm trước: -- USD")
                val worldChange = widgetData.getString("world_change", "0.00%")
                
                // Update text fields in XML layout
                setTextViewText(R.id.widget_title, title)
                setTextViewText(R.id.widget_updated_time, "Cập nhật: $updatedTime")
                setTextViewText(R.id.widget_world_price, worldPrice)
                setTextViewText(R.id.widget_yesterday_price, yesterdayPrice)
                setTextViewText(R.id.widget_world_change, worldChange)
                
                // Colorize the world change percentage
                if (worldChange?.startsWith("-") == true) {
                    setTextColor(R.id.widget_world_change, android.graphics.Color.parseColor("#F56565")) // Light red
                } else if (worldChange?.startsWith("+") == true || (worldChange != "0.00%" && worldChange != "--")) {
                    setTextColor(R.id.widget_world_change, android.graphics.Color.parseColor("#48BB78")) // Light green
                } else {
                    setTextColor(R.id.widget_world_change, android.graphics.Color.parseColor("#A0AEC0")) // Gray
                }

                // Set click PendingIntent to launch the app
                val pendingIntent = HomeWidgetLaunchIntent.getActivity(
                    context,
                    MainActivity::class.java
                )
                setOnClickPendingIntent(R.id.widget_container, pendingIntent)
            }
            appWidgetManager.updateAppWidget(appWidgetId, views)
        }
    }
}
